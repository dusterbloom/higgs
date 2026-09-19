use std::collections::{HashSet, VecDeque};

use super::{
    CDATA_CLOSE, CDATA_OPEN, FUNCTION_CLOSE, FUNCTION_OPEN, MAX_INSIDE_TOOL_CALL_BYTES,
    MINICPM_FUNCTION_OPEN, MINICPM_PARAM_CLOSE, MINICPM_PARAM_OPEN, PARAM_CLOSE, PARAM_OPEN,
    ParamType, TOOL_CALL_CLOSE, TOOL_CALL_OPEN, ToolSchema,
};

/// An ordered semantic event produced while model text is decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolStreamEvent {
    Text(String),
    ToolStart { index: usize, name: String },
    ArgumentsDelta { index: usize, fragment: String },
    ToolEnd { index: usize },
}

/// A terminal parse failure. Events returned beside the error remain valid;
/// tool argument fragments are append-only and are never replayed as prose.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolParseError {
    #[error("tool call exceeded the {limit}-byte limit")]
    CallTooLarge { limit: usize },
    #[error("model output ended inside a tool call")]
    IncompleteToolCall,
    #[error("malformed tool call: {message}")]
    Malformed { message: &'static str },
    #[error("duplicate argument key {name:?}")]
    DuplicateArgument { name: String },
    #[error("duplicate tool envelope field {name:?}")]
    DuplicateEnvelopeField { name: String },
    #[error("argument {name:?} is not a valid declared {expected}")]
    InvalidArgumentType {
        name: String,
        expected: &'static str,
    },
    #[error("JSON nesting exceeds the parser limit")]
    NestingTooDeep,
}

#[derive(Debug, Default)]
pub struct IncrementalToolCallOutput {
    pub events: Vec<ToolStreamEvent>,
    pub error: Option<ToolParseError>,
}

const MAX_JSON_NESTING: usize = 256;

/// Incrementally recognizes all tool syntaxes supported by the batch parser.
///
/// It retains only delimiter suffixes and values whose type is unknowable
/// without a schema; declared strings and JSON values stream append-only.
pub struct IncrementalToolCallTracker {
    active: bool,
    schema: Option<ToolSchema>,
    buffer: String,
    cursor: usize,
    state: State,
    call_bytes: usize,
    next_index: usize,
    completed_count: usize,
}

enum State {
    Scan,
    WrappedDetect,
    Legacy(Legacy),
    XmlName { dialect: XmlDialect, name: String },
    Xml(XmlCall),
    WrappedClose { index: usize },
    Failed,
    Poison,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum XmlDialect {
    Qwen,
    MiniCpm,
}

struct XmlCall {
    dialect: XmlDialect,
    index: usize,
    name: String,
    first_argument: bool,
    keys: HashSet<String>,
    phase: XmlPhase,
}

enum XmlPhase {
    BetweenArguments,
    ArgumentName(String),
    MiniArgumentTag(String),
    MiniValueProbe { key: String, probe: String },
    Value { key: String, value: XmlValue },
    AfterCdata,
}

struct XmlValue {
    scanner: DelimiterScanner,
    kind: XmlValueKind,
    strip_wrapping_newline: bool,
    leading: String,
    leading_done: bool,
    tail: VecDeque<char>,
}

enum XmlValueKind {
    String,
    Strict {
        expected: ParamType,
        validator: JsonValueTracker,
    },
    Buffered(String),
}

struct DelimiterScanner {
    delimiter: &'static str,
    pending: String,
}

struct Legacy {
    index: usize,
    phase: LegacyPhase,
    fields: HashSet<String>,
    name: Option<String>,
    started: bool,
    arguments_seen: bool,
    held_arguments: String,
}

enum LegacyPhase {
    KeyOrEnd {
        can_end: bool,
    },
    Key(JsonStringCapture),
    Colon(String),
    Value(String),
    Name(JsonStringCapture),
    JsonValue {
        field: String,
        validator: JsonValueTracker,
    },
    CommaOrEnd,
}

struct JsonStringCapture {
    raw: String,
    escaped: bool,
    unicode_digits: u8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RootKind {
    String,
    Number { integer: bool },
    Boolean,
    Null,
    Object,
    Array,
}

struct JsonValueTracker {
    mode: JsonMode,
    stack: Vec<JsonContainer>,
    root_kind: Option<RootKind>,
    complete: bool,
}

enum JsonMode {
    ExpectValue,
    ObjectKeyOrEnd,
    ObjectColon,
    ObjectCommaOrEnd,
    ArrayValueOrEnd,
    ArrayCommaOrEnd,
    String {
        role: StringRole,
        raw: String,
        escaped: bool,
        unicode_digits: u8,
    },
    Number(NumberState),
    Literal {
        expected: &'static str,
        position: usize,
    },
}

enum StringRole {
    Key,
    Value,
}

struct JsonContainer {
    kind: ContainerKind,
    keys: HashSet<String>,
    can_end: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ContainerKind {
    Object,
    Array,
}

#[derive(Clone, Copy)]
enum NumberState {
    Minus,
    Zero,
    Integer,
    Dot,
    Fraction,
    Exponent,
    ExponentSign,
    ExponentDigits,
}

enum JsonFeed {
    Consumed,
    CompleteConsumed,
    CompleteBefore,
}

impl IncrementalToolCallTracker {
    #[must_use]
    pub const fn new(active: bool, schema: Option<ToolSchema>) -> Self {
        Self {
            active,
            schema,
            buffer: String::new(),
            cursor: 0,
            state: State::Scan,
            call_bytes: 0,
            next_index: 0,
            completed_count: 0,
        }
    }

    #[must_use]
    pub const fn completed_call_count(&self) -> usize {
        self.completed_count
    }

    #[must_use]
    pub const fn holding(&self) -> bool {
        !matches!(self.state, State::Scan | State::Failed)
    }

    pub fn process(&mut self, text: &str) -> IncrementalToolCallOutput {
        if !self.active {
            return IncrementalToolCallOutput {
                events: (!text.is_empty())
                    .then(|| ToolStreamEvent::Text(text.to_owned()))
                    .into_iter()
                    .collect(),
                error: None,
            };
        }
        if matches!(self.state, State::Failed) {
            return IncrementalToolCallOutput::default();
        }

        self.buffer.push_str(text);
        let mut output = IncrementalToolCallOutput::default();
        loop {
            let state = std::mem::replace(&mut self.state, State::Poison);
            match self.step(state, &mut output.events) {
                Ok((next, progressed)) => {
                    self.state = next;
                    if !progressed {
                        break;
                    }
                }
                Err(error) => {
                    self.state = State::Failed;
                    self.buffer.clear();
                    self.cursor = 0;
                    output.error = Some(error);
                    break;
                }
            }
        }
        self.compact();
        output
    }

    pub fn finish(&mut self) -> IncrementalToolCallOutput {
        if !self.active || matches!(self.state, State::Failed) {
            return IncrementalToolCallOutput::default();
        }
        let mut output = IncrementalToolCallOutput::default();
        if matches!(self.state, State::Scan) {
            let leftover = self.available().to_owned();
            push_text(&mut output.events, leftover);
            self.cursor = self.buffer.len();
            self.compact();
        } else {
            self.state = State::Failed;
            self.buffer.clear();
            self.cursor = 0;
            output.error = Some(ToolParseError::IncompleteToolCall);
        }
        output
    }

    fn available(&self) -> &str {
        self.buffer.get(self.cursor..).unwrap_or_default()
    }

    fn compact(&mut self) {
        if self.cursor > 0 {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
    }

    const fn consume(&mut self, bytes: usize, inside_call: bool) -> Result<(), ToolParseError> {
        self.cursor += bytes;
        if inside_call {
            self.call_bytes += bytes;
            if self.call_bytes > MAX_INSIDE_TOOL_CALL_BYTES {
                return Err(ToolParseError::CallTooLarge {
                    limit: MAX_INSIDE_TOOL_CALL_BYTES,
                });
            }
        }
        Ok(())
    }

    fn step(
        &mut self,
        state: State,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        match state {
            State::Scan => self.step_scan(events),
            State::WrappedDetect => self.step_wrapped_detect(),
            State::Legacy(legacy) => self.step_legacy(legacy, events),
            State::XmlName { dialect, name } => self.step_xml_name(dialect, name, events),
            State::Xml(call) => self.step_xml(call, events),
            State::WrappedClose { index } => self.step_wrapped_close(index, events),
            State::Failed => Ok((State::Failed, false)),
            State::Poison => Err(ToolParseError::Malformed {
                message: "parser entered an invalid internal state",
            }),
        }
    }

    fn step_scan(
        &mut self,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        let available = self.available();
        if available.is_empty() {
            return Ok((State::Scan, false));
        }
        let wrapped = available
            .find(TOOL_CALL_OPEN)
            .map(|position| (position, true));
        let mini = available
            .find(MINICPM_FUNCTION_OPEN)
            .map(|position| (position, false));
        let opener = match (wrapped, mini) {
            (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
            (Some(found), None) | (None, Some(found)) => Some(found),
            (None, None) => None,
        };
        if let Some((position, is_wrapped)) = opener {
            push_text(events, available[..position].to_owned());
            self.consume(position, false)?;
            let opener_len = if is_wrapped {
                TOOL_CALL_OPEN.len()
            } else {
                MINICPM_FUNCTION_OPEN.len()
            };
            self.consume(opener_len, false)?;
            self.call_bytes = opener_len;
            let next = if is_wrapped {
                State::WrappedDetect
            } else {
                State::XmlName {
                    dialect: XmlDialect::MiniCpm,
                    name: String::new(),
                }
            };
            return Ok((next, true));
        }

        let keep = opener_suffix_len(available);
        let emit_len = available.len() - keep;
        if emit_len == 0 {
            return Ok((State::Scan, false));
        }
        push_text(events, available[..emit_len].to_owned());
        self.consume(emit_len, false)?;
        Ok((State::Scan, true))
    }

    fn step_wrapped_detect(&mut self) -> Result<(State, bool), ToolParseError> {
        let available = self.available();
        let whitespace = leading_whitespace_bytes(available);
        if whitespace > 0 {
            self.consume(whitespace, true)?;
            return Ok((State::WrappedDetect, true));
        }
        let remaining = self.available();
        if remaining.is_empty() {
            return Ok((State::WrappedDetect, false));
        }
        if remaining.starts_with('{') {
            self.consume(1, true)?;
            let index = self.next_index;
            self.next_index += 1;
            return Ok((State::Legacy(Legacy::new(index)), true));
        }
        if remaining.starts_with(FUNCTION_OPEN) {
            self.consume(FUNCTION_OPEN.len(), true)?;
            return Ok((
                State::XmlName {
                    dialect: XmlDialect::Qwen,
                    name: String::new(),
                },
                true,
            ));
        }
        if FUNCTION_OPEN.starts_with(remaining) {
            return Ok((State::WrappedDetect, false));
        }
        Err(ToolParseError::Malformed {
            message: "unsupported wrapped tool-call envelope",
        })
    }

    fn step_xml_name(
        &mut self,
        dialect: XmlDialect,
        mut name: String,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        let available = self.available();
        let Some(end) = available.find('>') else {
            if !available.is_empty() {
                name.push_str(available);
                let length = available.len();
                self.consume(length, true)?;
                return Ok((State::XmlName { dialect, name }, true));
            }
            return Ok((State::XmlName { dialect, name }, false));
        };
        name.push_str(&available[..end]);
        self.consume(end + 1, true)?;
        let parsed_name = match dialect {
            XmlDialect::Qwen => name.trim().to_owned(),
            XmlDialect::MiniCpm => parse_minicpm_name(&name)?,
        };
        if parsed_name.is_empty() {
            return Err(ToolParseError::Malformed {
                message: "tool name is empty",
            });
        }
        let index = self.next_index;
        self.next_index += 1;
        events.push(ToolStreamEvent::ToolStart {
            index,
            name: parsed_name.clone(),
        });
        push_arguments(events, index, "{".to_owned());
        Ok((
            State::Xml(XmlCall {
                dialect,
                index,
                name: parsed_name,
                first_argument: true,
                keys: HashSet::new(),
                phase: XmlPhase::BetweenArguments,
            }),
            true,
        ))
    }

    fn step_xml(
        &mut self,
        mut call: XmlCall,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        match std::mem::replace(&mut call.phase, XmlPhase::BetweenArguments) {
            XmlPhase::BetweenArguments => self.step_xml_between(call, events),
            XmlPhase::ArgumentName(name) => self.step_xml_argument_name(call, name, events),
            XmlPhase::MiniArgumentTag(tag) => self.step_mini_argument_tag(call, tag, events),
            XmlPhase::MiniValueProbe { key, probe } => {
                self.step_mini_value_probe(call, key, probe, events)
            }
            XmlPhase::Value { key, value } => self.step_xml_value(call, key, value, events),
            XmlPhase::AfterCdata => self.step_after_cdata(call),
        }
    }

    fn step_xml_between(
        &mut self,
        mut call: XmlCall,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        let whitespace = leading_whitespace_bytes(self.available());
        if whitespace > 0 {
            self.consume(whitespace, true)?;
            call.phase = XmlPhase::BetweenArguments;
            return Ok((State::Xml(call), true));
        }
        let available = self.available();
        if available.is_empty() {
            call.phase = XmlPhase::BetweenArguments;
            return Ok((State::Xml(call), false));
        }
        let argument_open = match call.dialect {
            XmlDialect::Qwen => PARAM_OPEN,
            XmlDialect::MiniCpm => MINICPM_PARAM_OPEN,
        };
        if available.starts_with(argument_open) {
            self.consume(argument_open.len(), true)?;
            call.phase = match call.dialect {
                XmlDialect::Qwen => XmlPhase::ArgumentName(String::new()),
                XmlDialect::MiniCpm => XmlPhase::MiniArgumentTag(String::new()),
            };
            return Ok((State::Xml(call), true));
        }
        if argument_open.starts_with(available) {
            call.phase = XmlPhase::BetweenArguments;
            return Ok((State::Xml(call), false));
        }
        if available.starts_with(FUNCTION_CLOSE) {
            self.consume(FUNCTION_CLOSE.len(), true)?;
            push_arguments(events, call.index, "}".to_owned());
            return match call.dialect {
                XmlDialect::Qwen => Ok((State::WrappedClose { index: call.index }, true)),
                XmlDialect::MiniCpm => {
                    self.complete_call(call.index, events);
                    Ok((State::Scan, true))
                }
            };
        }
        if FUNCTION_CLOSE.starts_with(available) {
            call.phase = XmlPhase::BetweenArguments;
            return Ok((State::Xml(call), false));
        }
        Err(ToolParseError::Malformed {
            message: "unexpected content between XML arguments",
        })
    }

    fn step_xml_argument_name(
        &mut self,
        mut call: XmlCall,
        mut name: String,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        let available = self.available();
        let Some(end) = available.find('>') else {
            if available.is_empty() {
                call.phase = XmlPhase::ArgumentName(name);
                return Ok((State::Xml(call), false));
            }
            name.push_str(available);
            let length = available.len();
            self.consume(length, true)?;
            call.phase = XmlPhase::ArgumentName(name);
            return Ok((State::Xml(call), true));
        };
        name.push_str(&available[..end]);
        self.consume(end + 1, true)?;
        let key = name.trim().to_owned();
        self.begin_xml_argument(&mut call, &key, events)?;
        call.phase = XmlPhase::Value {
            value: self.new_xml_value(&call, &key, PARAM_CLOSE, true),
            key,
        };
        Ok((State::Xml(call), true))
    }

    fn step_mini_argument_tag(
        &mut self,
        mut call: XmlCall,
        mut tag: String,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        let available = self.available();
        let Some(end) = available.find('>') else {
            if available.is_empty() {
                call.phase = XmlPhase::MiniArgumentTag(tag);
                return Ok((State::Xml(call), false));
            }
            tag.push_str(available);
            let length = available.len();
            self.consume(length, true)?;
            call.phase = XmlPhase::MiniArgumentTag(tag);
            return Ok((State::Xml(call), true));
        };
        tag.push_str(&available[..end]);
        self.consume(end + 1, true)?;
        let key_end = tag.find('"').ok_or(ToolParseError::Malformed {
            message: "unterminated MiniCPM argument name",
        })?;
        let key = tag[..key_end].to_owned();
        self.begin_xml_argument(&mut call, &key, events)?;
        call.phase = XmlPhase::MiniValueProbe {
            key,
            probe: String::new(),
        };
        Ok((State::Xml(call), true))
    }

    fn begin_xml_argument(
        &self,
        call: &mut XmlCall,
        key: &str,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(), ToolParseError> {
        if key.is_empty() {
            return Err(ToolParseError::Malformed {
                message: "argument name is empty",
            });
        }
        if !call.keys.insert(key.to_owned()) {
            return Err(ToolParseError::DuplicateArgument {
                name: key.to_owned(),
            });
        }
        let mut fragment = String::new();
        if !call.first_argument {
            fragment.push(',');
        }
        call.first_argument = false;
        fragment.push_str(&serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_owned()));
        fragment.push(':');
        let declared = self
            .schema
            .as_ref()
            .and_then(|schema| schema.param_type(&call.name, key));
        if declared == Some(ParamType::Str) {
            fragment.push('"');
        }
        push_arguments(events, call.index, fragment);
        Ok(())
    }

    fn new_xml_value(
        &self,
        call: &XmlCall,
        key: &str,
        delimiter: &'static str,
        strip_wrapping_newline: bool,
    ) -> XmlValue {
        let declared = self
            .schema
            .as_ref()
            .and_then(|schema| schema.param_type(&call.name, key));
        XmlValue::new(delimiter, declared, strip_wrapping_newline)
    }

    fn step_mini_value_probe(
        &mut self,
        mut call: XmlCall,
        key: String,
        mut probe: String,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        let available = self.available();
        if available.is_empty() {
            call.phase = XmlPhase::MiniValueProbe { key, probe };
            return Ok((State::Xml(call), false));
        }
        let character = available.chars().next().unwrap_or_default();
        probe.push(character);
        self.consume(character.len_utf8(), true)?;
        if probe == CDATA_OPEN {
            call.phase = XmlPhase::Value {
                value: self.new_xml_value(&call, &key, CDATA_CLOSE, false),
                key,
            };
            return Ok((State::Xml(call), true));
        }
        if CDATA_OPEN.starts_with(&probe) {
            call.phase = XmlPhase::MiniValueProbe { key, probe };
            return Ok((State::Xml(call), true));
        }

        let mut value = self.new_xml_value(&call, &key, MINICPM_PARAM_CLOSE, false);
        let mut fragment = String::new();
        let matched = value.feed(&probe, &mut fragment)?;
        if matched {
            value.finish(&key, &mut fragment)?;
            call.phase = XmlPhase::BetweenArguments;
        } else {
            call.phase = XmlPhase::Value { key, value };
        }
        push_arguments(events, call.index, fragment);
        Ok((State::Xml(call), true))
    }

    fn step_xml_value(
        &mut self,
        mut call: XmlCall,
        key: String,
        mut value: XmlValue,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        let available = self.available();
        if available.is_empty() {
            call.phase = XmlPhase::Value { key, value };
            return Ok((State::Xml(call), false));
        }
        let mut fragment = String::new();
        let mut consumed = 0;
        let mut matched = false;
        for character in available.chars() {
            consumed += character.len_utf8();
            if value.feed_char(character, &mut fragment)? {
                matched = true;
                break;
            }
        }
        self.consume(consumed, true)?;
        if matched {
            value.finish(&key, &mut fragment)?;
        }
        push_arguments(events, call.index, fragment);
        if matched {
            call.phase = if value.scanner.delimiter == CDATA_CLOSE {
                XmlPhase::AfterCdata
            } else {
                XmlPhase::BetweenArguments
            };
        } else {
            call.phase = XmlPhase::Value { key, value };
        }
        Ok((State::Xml(call), true))
    }

    fn step_after_cdata(&mut self, mut call: XmlCall) -> Result<(State, bool), ToolParseError> {
        let whitespace = leading_whitespace_bytes(self.available());
        if whitespace > 0 {
            self.consume(whitespace, true)?;
            call.phase = XmlPhase::AfterCdata;
            return Ok((State::Xml(call), true));
        }
        let available = self.available();
        if available.starts_with(MINICPM_PARAM_CLOSE) {
            self.consume(MINICPM_PARAM_CLOSE.len(), true)?;
            call.phase = XmlPhase::BetweenArguments;
            return Ok((State::Xml(call), true));
        }
        if MINICPM_PARAM_CLOSE.starts_with(available) {
            call.phase = XmlPhase::AfterCdata;
            return Ok((State::Xml(call), false));
        }
        Err(ToolParseError::Malformed {
            message: "CDATA is not followed by </param>",
        })
    }

    fn step_legacy(
        &mut self,
        mut legacy: Legacy,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        match std::mem::replace(&mut legacy.phase, LegacyPhase::KeyOrEnd { can_end: false }) {
            LegacyPhase::KeyOrEnd { can_end } => {
                self.step_legacy_key_or_end(legacy, can_end, events)
            }
            LegacyPhase::Key(capture) => self.step_legacy_capture(legacy, capture, true, events),
            LegacyPhase::Colon(field) => self.step_legacy_colon(legacy, field),
            LegacyPhase::Value(field) => self.step_legacy_value(legacy, field),
            LegacyPhase::Name(capture) => self.step_legacy_capture(legacy, capture, false, events),
            LegacyPhase::JsonValue { field, validator } => {
                self.step_legacy_json(legacy, field, validator, events)
            }
            LegacyPhase::CommaOrEnd => self.step_legacy_comma_or_end(legacy, events),
        }
    }

    fn step_legacy_key_or_end(
        &mut self,
        mut legacy: Legacy,
        can_end: bool,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        if self.consume_leading_call_whitespace()? {
            legacy.phase = LegacyPhase::KeyOrEnd { can_end };
            return Ok((State::Legacy(legacy), true));
        }
        let available = self.available();
        if available.is_empty() {
            legacy.phase = LegacyPhase::KeyOrEnd { can_end };
            return Ok((State::Legacy(legacy), false));
        }
        if available.starts_with('}') {
            if !can_end {
                return Err(ToolParseError::Malformed {
                    message: "trailing comma in legacy envelope",
                });
            }
            self.consume(1, true)?;
            return Self::finish_legacy_envelope(legacy, events);
        }
        if available.starts_with('"') {
            self.consume(1, true)?;
            legacy.phase = LegacyPhase::Key(JsonStringCapture::new());
            return Ok((State::Legacy(legacy), true));
        }
        Err(ToolParseError::Malformed {
            message: "legacy envelope expected a field name",
        })
    }

    fn step_legacy_capture(
        &mut self,
        mut legacy: Legacy,
        mut capture: JsonStringCapture,
        is_key: bool,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        let available = self.available();
        if available.is_empty() {
            legacy.phase = if is_key {
                LegacyPhase::Key(capture)
            } else {
                LegacyPhase::Name(capture)
            };
            return Ok((State::Legacy(legacy), false));
        }
        let mut consumed = 0;
        let mut complete = false;
        for character in available.chars() {
            consumed += character.len_utf8();
            if capture.feed(character)? {
                complete = true;
                break;
            }
        }
        self.consume(consumed, true)?;
        if !complete {
            legacy.phase = if is_key {
                LegacyPhase::Key(capture)
            } else {
                LegacyPhase::Name(capture)
            };
            return Ok((State::Legacy(legacy), true));
        }
        let decoded = capture.decode()?;
        if is_key {
            if !legacy.fields.insert(decoded.clone()) {
                return Err(ToolParseError::DuplicateEnvelopeField { name: decoded });
            }
            legacy.phase = LegacyPhase::Colon(decoded);
        } else {
            legacy.name = Some(decoded);
            Self::start_legacy_if_ready(&mut legacy, events);
            legacy.phase = LegacyPhase::CommaOrEnd;
        }
        Ok((State::Legacy(legacy), true))
    }

    fn step_legacy_colon(
        &mut self,
        mut legacy: Legacy,
        field: String,
    ) -> Result<(State, bool), ToolParseError> {
        if self.consume_leading_call_whitespace()? {
            legacy.phase = LegacyPhase::Colon(field);
            return Ok((State::Legacy(legacy), true));
        }
        let available = self.available();
        if available.is_empty() {
            legacy.phase = LegacyPhase::Colon(field);
            return Ok((State::Legacy(legacy), false));
        }
        if !available.starts_with(':') {
            return Err(ToolParseError::Malformed {
                message: "legacy envelope field is missing ':'",
            });
        }
        self.consume(1, true)?;
        legacy.phase = LegacyPhase::Value(field);
        Ok((State::Legacy(legacy), true))
    }

    fn step_legacy_value(
        &mut self,
        mut legacy: Legacy,
        field: String,
    ) -> Result<(State, bool), ToolParseError> {
        if self.consume_leading_call_whitespace()? {
            legacy.phase = LegacyPhase::Value(field);
            return Ok((State::Legacy(legacy), true));
        }
        let available = self.available();
        if available.is_empty() {
            legacy.phase = LegacyPhase::Value(field);
            return Ok((State::Legacy(legacy), false));
        }
        if field == "name" {
            if !available.starts_with('"') {
                return Err(ToolParseError::Malformed {
                    message: "legacy tool name must be a string",
                });
            }
            self.consume(1, true)?;
            legacy.phase = LegacyPhase::Name(JsonStringCapture::new());
        } else {
            if field == "arguments" {
                legacy.arguments_seen = true;
            }
            legacy.phase = LegacyPhase::JsonValue {
                field,
                validator: JsonValueTracker::new(),
            };
        }
        Ok((State::Legacy(legacy), true))
    }

    fn step_legacy_json(
        &mut self,
        mut legacy: Legacy,
        field: String,
        mut validator: JsonValueTracker,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        let available = self.available();
        if available.is_empty() {
            legacy.phase = LegacyPhase::JsonValue { field, validator };
            return Ok((State::Legacy(legacy), false));
        }
        let is_arguments = field == "arguments";
        let mut fragment = String::new();
        let mut consumed = 0;
        let mut complete = false;
        for character in available.chars() {
            match validator.feed(character)? {
                JsonFeed::CompleteBefore => {
                    complete = true;
                    break;
                }
                JsonFeed::Consumed => {
                    consumed += character.len_utf8();
                    if is_arguments {
                        fragment.push(character);
                    }
                }
                JsonFeed::CompleteConsumed => {
                    consumed += character.len_utf8();
                    if is_arguments {
                        fragment.push(character);
                    }
                    complete = true;
                    break;
                }
            }
        }
        self.consume(consumed, true)?;
        if is_arguments && !fragment.is_empty() {
            if legacy.started {
                push_arguments(events, legacy.index, fragment);
            } else {
                legacy.held_arguments.push_str(&fragment);
            }
        }
        if complete {
            legacy.phase = LegacyPhase::CommaOrEnd;
        } else {
            legacy.phase = LegacyPhase::JsonValue { field, validator };
        }
        Ok((State::Legacy(legacy), consumed > 0 || complete))
    }

    fn step_legacy_comma_or_end(
        &mut self,
        mut legacy: Legacy,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        if self.consume_leading_call_whitespace()? {
            legacy.phase = LegacyPhase::CommaOrEnd;
            return Ok((State::Legacy(legacy), true));
        }
        let available = self.available();
        if available.is_empty() {
            legacy.phase = LegacyPhase::CommaOrEnd;
            return Ok((State::Legacy(legacy), false));
        }
        if available.starts_with(',') {
            self.consume(1, true)?;
            legacy.phase = LegacyPhase::KeyOrEnd { can_end: false };
            return Ok((State::Legacy(legacy), true));
        }
        if available.starts_with('}') {
            self.consume(1, true)?;
            return Self::finish_legacy_envelope(legacy, events);
        }
        Err(ToolParseError::Malformed {
            message: "legacy envelope expected ',' or '}'",
        })
    }

    fn finish_legacy_envelope(
        mut legacy: Legacy,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        let Some(_) = legacy.name else {
            return Err(ToolParseError::Malformed {
                message: "legacy envelope has no tool name",
            });
        };
        Self::start_legacy_if_ready(&mut legacy, events);
        if !legacy.arguments_seen {
            push_arguments(events, legacy.index, "{}".to_owned());
        }
        Ok((
            State::WrappedClose {
                index: legacy.index,
            },
            true,
        ))
    }

    fn start_legacy_if_ready(legacy: &mut Legacy, events: &mut Vec<ToolStreamEvent>) {
        if legacy.started {
            return;
        }
        let Some(name) = legacy.name.as_ref() else {
            return;
        };
        legacy.started = true;
        events.push(ToolStreamEvent::ToolStart {
            index: legacy.index,
            name: name.clone(),
        });
        if !legacy.held_arguments.is_empty() {
            push_arguments(
                events,
                legacy.index,
                std::mem::take(&mut legacy.held_arguments),
            );
        }
    }

    fn step_wrapped_close(
        &mut self,
        index: usize,
        events: &mut Vec<ToolStreamEvent>,
    ) -> Result<(State, bool), ToolParseError> {
        let whitespace = leading_whitespace_bytes(self.available());
        if whitespace > 0 {
            self.consume(whitespace, true)?;
            return Ok((State::WrappedClose { index }, true));
        }
        let available = self.available();
        if available.starts_with(TOOL_CALL_CLOSE) {
            self.consume(TOOL_CALL_CLOSE.len(), true)?;
            self.complete_call(index, events);
            return Ok((State::Scan, true));
        }
        if TOOL_CALL_CLOSE.starts_with(available) {
            return Ok((State::WrappedClose { index }, false));
        }
        Err(ToolParseError::Malformed {
            message: "wrapped call is missing </tool_call>",
        })
    }

    fn complete_call(&mut self, index: usize, events: &mut Vec<ToolStreamEvent>) {
        events.push(ToolStreamEvent::ToolEnd { index });
        self.completed_count += 1;
        self.call_bytes = 0;
    }

    fn consume_leading_call_whitespace(&mut self) -> Result<bool, ToolParseError> {
        let count = leading_json_whitespace_bytes(self.available());
        if count == 0 {
            return Ok(false);
        }
        self.consume(count, true)?;
        Ok(true)
    }
}

impl Legacy {
    fn new(index: usize) -> Self {
        Self {
            index,
            phase: LegacyPhase::KeyOrEnd { can_end: true },
            fields: HashSet::new(),
            name: None,
            started: false,
            arguments_seen: false,
            held_arguments: String::new(),
        }
    }
}

impl DelimiterScanner {
    const fn new(delimiter: &'static str) -> Self {
        Self {
            delimiter,
            pending: String::new(),
        }
    }

    fn feed(&mut self, character: char, proven: &mut String) -> bool {
        self.pending.push(character);
        if self.pending == self.delimiter {
            self.pending.clear();
            return true;
        }
        if self.delimiter.starts_with(&self.pending) {
            return false;
        }

        let mut suffix_start = self.pending.len();
        for (index, _) in self.pending.char_indices().skip(1) {
            if self.delimiter.starts_with(&self.pending[index..]) {
                suffix_start = index;
                break;
            }
        }
        if suffix_start == self.pending.len() {
            proven.push_str(&self.pending);
            self.pending.clear();
        } else {
            proven.push_str(&self.pending[..suffix_start]);
            self.pending.drain(..suffix_start);
        }
        false
    }
}

impl XmlValue {
    const fn new(
        delimiter: &'static str,
        declared: Option<ParamType>,
        strip_wrapping_newline: bool,
    ) -> Self {
        let kind = match declared {
            Some(ParamType::Str) => XmlValueKind::String,
            Some(expected) => XmlValueKind::Strict {
                expected,
                validator: JsonValueTracker::new(),
            },
            None => XmlValueKind::Buffered(String::new()),
        };
        Self {
            scanner: DelimiterScanner::new(delimiter),
            kind,
            strip_wrapping_newline,
            leading: String::new(),
            leading_done: !strip_wrapping_newline,
            tail: VecDeque::new(),
        }
    }

    fn feed(&mut self, input: &str, output: &mut String) -> Result<bool, ToolParseError> {
        for character in input.chars() {
            if self.feed_char(character, output)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn feed_char(&mut self, character: char, output: &mut String) -> Result<bool, ToolParseError> {
        if matches!(self.kind, XmlValueKind::Strict { .. }) {
            return self.feed_strict_char(character, output);
        }
        let mut proven = String::new();
        let matched = self.scanner.feed(character, &mut proven);
        if !proven.is_empty() {
            self.consume_proven(&proven, output)?;
        }
        Ok(matched)
    }

    fn feed_strict_char(
        &mut self,
        character: char,
        output: &mut String,
    ) -> Result<bool, ToolParseError> {
        if !self.leading_done {
            self.leading.push(character);
            match self.leading.as_str() {
                "\n" | "\r\n" => {
                    self.leading.clear();
                    self.leading_done = true;
                    return Ok(false);
                }
                "\r" => return Ok(false),
                _ => {
                    let leading = std::mem::take(&mut self.leading);
                    self.leading_done = true;
                    for leading_character in leading.chars() {
                        if self.feed_strict_content(leading_character, output)? {
                            return Ok(true);
                        }
                    }
                    return Ok(false);
                }
            }
        }
        self.feed_strict_content(character, output)
    }

    fn feed_strict_content(
        &mut self,
        character: char,
        output: &mut String,
    ) -> Result<bool, ToolParseError> {
        let XmlValueKind::Strict { validator, .. } = &mut self.kind else {
            return Err(ToolParseError::Malformed {
                message: "strict XML parser entered an invalid internal state",
            });
        };
        if !validator.complete {
            match validator.feed(character)? {
                JsonFeed::Consumed | JsonFeed::CompleteConsumed => {
                    output.push(character);
                    return Ok(false);
                }
                JsonFeed::CompleteBefore => {}
            }
        }

        let mut proven = String::new();
        let matched = self.scanner.feed(character, &mut proven);
        for suffix_character in proven.chars() {
            if !is_json_whitespace(suffix_character) {
                return Err(ToolParseError::Malformed {
                    message: "characters follow a complete typed argument",
                });
            }
            self.push_strict_suffix(suffix_character, output);
        }
        Ok(matched)
    }

    fn push_strict_suffix(&mut self, character: char, output: &mut String) {
        match character {
            '\n' if self.tail.back() == Some(&'\r') => self.tail.push_back(character),
            '\n' | '\r' => {
                output.extend(self.tail.drain(..));
                self.tail.push_back(character);
            }
            _ => {
                output.extend(self.tail.drain(..));
                output.push(character);
            }
        }
    }

    fn consume_proven(&mut self, proven: &str, output: &mut String) -> Result<(), ToolParseError> {
        for character in proven.chars() {
            if self.leading_done {
                self.push_tail(character, output)?;
                continue;
            }
            self.leading.push(character);
            match self.leading.as_str() {
                "\n" | "\r\n" => {
                    self.leading.clear();
                    self.leading_done = true;
                }
                "\r" => {}
                _ => {
                    let leading = std::mem::take(&mut self.leading);
                    self.leading_done = true;
                    for leading_character in leading.chars() {
                        self.push_tail(leading_character, output)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn push_tail(&mut self, character: char, output: &mut String) -> Result<(), ToolParseError> {
        if !self.strip_wrapping_newline {
            return self.emit_character(character, output);
        }

        match character {
            '\n' if self.tail.back() == Some(&'\r') => self.tail.push_back(character),
            '\n' | '\r' => {
                while let Some(ready) = self.tail.pop_front() {
                    self.emit_character(ready, output)?;
                }
                self.tail.push_back(character);
            }
            _ => {
                while let Some(ready) = self.tail.pop_front() {
                    self.emit_character(ready, output)?;
                }
                self.emit_character(character, output)?;
            }
        }
        Ok(())
    }

    fn finish(&mut self, key: &str, output: &mut String) -> Result<(), ToolParseError> {
        if !self.leading_done {
            let leading = std::mem::take(&mut self.leading);
            self.leading_done = true;
            for character in leading.chars() {
                self.tail.push_back(character);
            }
        }
        if self.strip_wrapping_newline {
            if self.tail.back() == Some(&'\n') {
                self.tail.pop_back();
                if self.tail.back() == Some(&'\r') {
                    self.tail.pop_back();
                }
            }
        }
        if matches!(self.kind, XmlValueKind::Strict { .. }) {
            while let Some(character) = self.tail.pop_front() {
                output.push(character);
            }
        } else {
            while let Some(character) = self.tail.pop_front() {
                self.emit_character(character, output)?;
            }
        }
        match &mut self.kind {
            XmlValueKind::String => output.push('"'),
            XmlValueKind::Strict {
                expected,
                validator,
            } => {
                validator.finish()?;
                if !root_matches(*expected, validator.root_kind) {
                    return Err(ToolParseError::InvalidArgumentType {
                        name: key.to_owned(),
                        expected: param_type_name(*expected),
                    });
                }
            }
            XmlValueKind::Buffered(raw) => {
                let value = super::coerce_param_value(raw, None);
                output.push_str(&serde_json::to_string(&value).map_err(|_| {
                    ToolParseError::Malformed {
                        message: "argument could not be encoded as JSON",
                    }
                })?);
            }
        }
        Ok(())
    }

    fn emit_character(
        &mut self,
        character: char,
        output: &mut String,
    ) -> Result<(), ToolParseError> {
        match &mut self.kind {
            XmlValueKind::String => escape_json_character(character, output),
            XmlValueKind::Strict { validator, .. } => match validator.feed(character)? {
                JsonFeed::Consumed | JsonFeed::CompleteConsumed => output.push(character),
                JsonFeed::CompleteBefore if is_json_whitespace(character) => {
                    output.push(character);
                }
                JsonFeed::CompleteBefore => {
                    return Err(ToolParseError::Malformed {
                        message: "characters follow a complete typed argument",
                    });
                }
            },
            XmlValueKind::Buffered(raw) => raw.push(character),
        }
        Ok(())
    }
}

impl JsonStringCapture {
    fn new() -> Self {
        Self {
            raw: "\"".to_owned(),
            escaped: false,
            unicode_digits: 0,
        }
    }

    fn feed(&mut self, character: char) -> Result<bool, ToolParseError> {
        self.raw.push(character);
        if self.unicode_digits > 0 {
            if !character.is_ascii_hexdigit() {
                return Err(ToolParseError::Malformed {
                    message: "invalid JSON unicode escape",
                });
            }
            self.unicode_digits -= 1;
            return Ok(false);
        }
        if self.escaped {
            self.escaped = false;
            if character == 'u' {
                self.unicode_digits = 4;
            } else if !matches!(character, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't') {
                return Err(ToolParseError::Malformed {
                    message: "invalid JSON escape",
                });
            }
            return Ok(false);
        }
        match character {
            '\\' => self.escaped = true,
            '"' => return Ok(true),
            value if value <= '\u{1f}' => {
                return Err(ToolParseError::Malformed {
                    message: "unescaped control character in JSON string",
                });
            }
            _ => {}
        }
        Ok(false)
    }

    fn decode(self) -> Result<String, ToolParseError> {
        serde_json::from_str(&self.raw).map_err(|_| ToolParseError::Malformed {
            message: "invalid JSON string",
        })
    }
}

impl JsonValueTracker {
    const fn new() -> Self {
        Self {
            mode: JsonMode::ExpectValue,
            stack: Vec::new(),
            root_kind: None,
            complete: false,
        }
    }

    #[allow(clippy::too_many_lines)] // The JSON DFA stays in one exhaustive state transition.
    fn feed(&mut self, character: char) -> Result<JsonFeed, ToolParseError> {
        if self.complete {
            return if is_json_whitespace(character) || matches!(character, ',' | '}') {
                Ok(JsonFeed::CompleteBefore)
            } else {
                Err(ToolParseError::Malformed {
                    message: "characters follow a complete JSON value",
                })
            };
        }
        loop {
            match &mut self.mode {
                JsonMode::ExpectValue | JsonMode::ArrayValueOrEnd => {
                    let array_can_end = matches!(self.mode, JsonMode::ArrayValueOrEnd)
                        && self.stack.last().is_some_and(|frame| frame.can_end);
                    if is_json_whitespace(character) {
                        return Ok(JsonFeed::Consumed);
                    }
                    if array_can_end && character == ']' {
                        self.close_container(ContainerKind::Array)?;
                        return Ok(self.consumed_status());
                    }
                    match character {
                        '{' => self.open_container(ContainerKind::Object)?,
                        '[' => self.open_container(ContainerKind::Array)?,
                        '"' => {
                            self.set_root_kind(RootKind::String);
                            self.mode = JsonMode::String {
                                role: StringRole::Value,
                                raw: "\"".to_owned(),
                                escaped: false,
                                unicode_digits: 0,
                            };
                        }
                        't' => self.start_literal("true", RootKind::Boolean),
                        'f' => self.start_literal("false", RootKind::Boolean),
                        'n' => self.start_literal("null", RootKind::Null),
                        '-' => {
                            self.set_root_kind(RootKind::Number { integer: true });
                            self.mode = JsonMode::Number(NumberState::Minus);
                        }
                        '0' => {
                            self.set_root_kind(RootKind::Number { integer: true });
                            self.mode = JsonMode::Number(NumberState::Zero);
                        }
                        '1'..='9' => {
                            self.set_root_kind(RootKind::Number { integer: true });
                            self.mode = JsonMode::Number(NumberState::Integer);
                        }
                        _ => {
                            return Err(ToolParseError::Malformed {
                                message: "invalid JSON value",
                            });
                        }
                    }
                    return Ok(self.consumed_status());
                }
                JsonMode::ObjectKeyOrEnd => {
                    if is_json_whitespace(character) {
                        return Ok(JsonFeed::Consumed);
                    }
                    if character == '}' {
                        if !self.stack.last().is_some_and(|frame| frame.can_end) {
                            return Err(ToolParseError::Malformed {
                                message: "trailing comma in JSON object",
                            });
                        }
                        self.close_container(ContainerKind::Object)?;
                        return Ok(self.consumed_status());
                    }
                    if character != '"' {
                        return Err(ToolParseError::Malformed {
                            message: "JSON object key must be a string",
                        });
                    }
                    self.mode = JsonMode::String {
                        role: StringRole::Key,
                        raw: "\"".to_owned(),
                        escaped: false,
                        unicode_digits: 0,
                    };
                    return Ok(JsonFeed::Consumed);
                }
                JsonMode::ObjectColon => {
                    if is_json_whitespace(character) {
                        return Ok(JsonFeed::Consumed);
                    }
                    if character != ':' {
                        return Err(ToolParseError::Malformed {
                            message: "JSON object key is missing ':'",
                        });
                    }
                    self.mode = JsonMode::ExpectValue;
                    return Ok(JsonFeed::Consumed);
                }
                JsonMode::ObjectCommaOrEnd => {
                    if is_json_whitespace(character) {
                        return Ok(JsonFeed::Consumed);
                    }
                    self.mode = match character {
                        ',' => {
                            if let Some(frame) = self.stack.last_mut() {
                                frame.can_end = false;
                            }
                            JsonMode::ObjectKeyOrEnd
                        }
                        '}' => {
                            self.close_container(ContainerKind::Object)?;
                            return Ok(self.consumed_status());
                        }
                        _ => {
                            return Err(ToolParseError::Malformed {
                                message: "JSON object expected ',' or '}'",
                            });
                        }
                    };
                    return Ok(JsonFeed::Consumed);
                }
                JsonMode::ArrayCommaOrEnd => {
                    if is_json_whitespace(character) {
                        return Ok(JsonFeed::Consumed);
                    }
                    self.mode = match character {
                        ',' => {
                            if let Some(frame) = self.stack.last_mut() {
                                frame.can_end = false;
                            }
                            JsonMode::ArrayValueOrEnd
                        }
                        ']' => {
                            self.close_container(ContainerKind::Array)?;
                            return Ok(self.consumed_status());
                        }
                        _ => {
                            return Err(ToolParseError::Malformed {
                                message: "JSON array expected ',' or ']'",
                            });
                        }
                    };
                    return Ok(JsonFeed::Consumed);
                }
                JsonMode::String {
                    role,
                    raw,
                    escaped,
                    unicode_digits,
                } => {
                    raw.push(character);
                    if *unicode_digits > 0 {
                        if !character.is_ascii_hexdigit() {
                            return Err(ToolParseError::Malformed {
                                message: "invalid JSON unicode escape",
                            });
                        }
                        *unicode_digits -= 1;
                        return Ok(JsonFeed::Consumed);
                    }
                    if *escaped {
                        *escaped = false;
                        if character == 'u' {
                            *unicode_digits = 4;
                        } else if !matches!(
                            character,
                            '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't'
                        ) {
                            return Err(ToolParseError::Malformed {
                                message: "invalid JSON escape",
                            });
                        }
                        return Ok(JsonFeed::Consumed);
                    }
                    match character {
                        '\\' => *escaped = true,
                        '"' => {
                            let completed_role = std::mem::replace(role, StringRole::Value);
                            if matches!(completed_role, StringRole::Key) {
                                let key: String = serde_json::from_str(raw).map_err(|_| {
                                    ToolParseError::Malformed {
                                        message: "invalid JSON object key",
                                    }
                                })?;
                                let frame =
                                    self.stack.last_mut().ok_or(ToolParseError::Malformed {
                                        message: "JSON key outside object",
                                    })?;
                                if !frame.keys.insert(key.clone()) {
                                    return Err(ToolParseError::DuplicateArgument { name: key });
                                }
                                self.mode = JsonMode::ObjectColon;
                            } else {
                                self.value_finished();
                            }
                        }
                        value if value <= '\u{1f}' => {
                            return Err(ToolParseError::Malformed {
                                message: "unescaped control character in JSON string",
                            });
                        }
                        _ => {}
                    }
                    return Ok(self.consumed_status());
                }
                JsonMode::Number(state) => {
                    if advance_number(state, character)? {
                        if matches!(character, '.' | 'e' | 'E') {
                            self.root_kind = Some(RootKind::Number { integer: false });
                        }
                        return Ok(JsonFeed::Consumed);
                    }
                    if number_can_end(*state) {
                        self.value_finished();
                        if self.complete {
                            return Ok(JsonFeed::CompleteBefore);
                        }
                        continue;
                    }
                    return Err(ToolParseError::Malformed {
                        message: "invalid JSON number",
                    });
                }
                JsonMode::Literal { expected, position } => {
                    let expected_character =
                        expected
                            .chars()
                            .nth(*position)
                            .ok_or(ToolParseError::Malformed {
                                message: "invalid JSON literal",
                            })?;
                    if character != expected_character {
                        return Err(ToolParseError::Malformed {
                            message: "invalid JSON literal",
                        });
                    }
                    *position += 1;
                    if *position == expected.len() {
                        self.value_finished();
                    }
                    return Ok(self.consumed_status());
                }
            }
        }
    }

    fn finish(&mut self) -> Result<(), ToolParseError> {
        if self.complete {
            return Ok(());
        }
        if let JsonMode::Number(state) = self.mode {
            if number_can_end(state) {
                self.value_finished();
            }
        }
        if self.complete {
            Ok(())
        } else {
            Err(ToolParseError::Malformed {
                message: "incomplete JSON value",
            })
        }
    }

    const fn set_root_kind(&mut self, kind: RootKind) {
        if self.stack.is_empty() && self.root_kind.is_none() {
            self.root_kind = Some(kind);
        }
    }

    fn start_literal(&mut self, expected: &'static str, kind: RootKind) {
        self.set_root_kind(kind);
        self.mode = JsonMode::Literal {
            expected,
            position: 1,
        };
    }

    fn open_container(&mut self, kind: ContainerKind) -> Result<(), ToolParseError> {
        if self.stack.len() >= MAX_JSON_NESTING {
            return Err(ToolParseError::NestingTooDeep);
        }
        self.set_root_kind(match kind {
            ContainerKind::Object => RootKind::Object,
            ContainerKind::Array => RootKind::Array,
        });
        self.stack.push(JsonContainer {
            kind,
            keys: HashSet::new(),
            can_end: true,
        });
        self.mode = match kind {
            ContainerKind::Object => JsonMode::ObjectKeyOrEnd,
            ContainerKind::Array => JsonMode::ArrayValueOrEnd,
        };
        Ok(())
    }

    fn close_container(&mut self, expected: ContainerKind) -> Result<(), ToolParseError> {
        let frame = self.stack.pop().ok_or(ToolParseError::Malformed {
            message: "unexpected JSON container close",
        })?;
        if frame.kind != expected {
            return Err(ToolParseError::Malformed {
                message: "mismatched JSON container close",
            });
        }
        self.value_finished();
        Ok(())
    }

    fn value_finished(&mut self) {
        let Some(parent) = self.stack.last_mut() else {
            self.complete = true;
            return;
        };
        parent.can_end = true;
        self.mode = match parent.kind {
            ContainerKind::Object => JsonMode::ObjectCommaOrEnd,
            ContainerKind::Array => JsonMode::ArrayCommaOrEnd,
        };
    }

    const fn consumed_status(&self) -> JsonFeed {
        if self.complete {
            JsonFeed::CompleteConsumed
        } else {
            JsonFeed::Consumed
        }
    }
}

const fn advance_number(state: &mut NumberState, character: char) -> Result<bool, ToolParseError> {
    use NumberState::{
        Dot, Exponent, ExponentDigits, ExponentSign, Fraction, Integer, Minus, Zero,
    };
    let next = match (*state, character) {
        (Minus, '0') => Some(Zero),
        (Minus, '1'..='9') | (Integer, '0'..='9') => Some(Integer),
        (Zero | Integer, '.') => Some(Dot),
        (Zero | Integer | Fraction, 'e' | 'E') => Some(Exponent),
        (Dot | Fraction, '0'..='9') => Some(Fraction),
        (Exponent, '+' | '-') => Some(ExponentSign),
        (Exponent | ExponentSign | ExponentDigits, '0'..='9') => Some(ExponentDigits),
        (Zero, '0'..='9') => {
            return Err(ToolParseError::Malformed {
                message: "JSON number has a leading zero",
            });
        }
        _ => None,
    };
    match next {
        Some(next_state) => {
            *state = next_state;
            Ok(true)
        }
        None => Ok(false),
    }
}

const fn number_can_end(state: NumberState) -> bool {
    matches!(
        state,
        NumberState::Zero
            | NumberState::Integer
            | NumberState::Fraction
            | NumberState::ExponentDigits
    )
}

const fn root_matches(expected: ParamType, actual: Option<RootKind>) -> bool {
    matches!(
        (expected, actual),
        (ParamType::Integer, Some(RootKind::Number { integer: true }))
            | (ParamType::Number, Some(RootKind::Number { .. }))
            | (ParamType::Boolean, Some(RootKind::Boolean))
            | (ParamType::Object, Some(RootKind::Object))
            | (ParamType::Array, Some(RootKind::Array))
            | (ParamType::Str, Some(RootKind::String))
    )
}

const fn param_type_name(param_type: ParamType) -> &'static str {
    match param_type {
        ParamType::Str => "string",
        ParamType::Integer => "integer",
        ParamType::Number => "number",
        ParamType::Boolean => "boolean",
        ParamType::Object => "object",
        ParamType::Array => "array",
    }
}

fn parse_minicpm_name(tag_remainder: &str) -> Result<String, ToolParseError> {
    let name_start = tag_remainder
        .find("name=\"")
        .ok_or(ToolParseError::Malformed {
            message: "MiniCPM function has no name attribute",
        })?
        + "name=\"".len();
    let remainder = &tag_remainder[name_start..];
    let end = remainder.find('"').ok_or(ToolParseError::Malformed {
        message: "unterminated MiniCPM function name",
    })?;
    Ok(remainder[..end].to_owned())
}

fn opener_suffix_len(input: &str) -> usize {
    let mut best = 0;
    for (index, _) in input.char_indices() {
        let suffix = &input[index..];
        if TOOL_CALL_OPEN.starts_with(suffix) || MINICPM_FUNCTION_OPEN.starts_with(suffix) {
            best = best.max(suffix.len());
        }
    }
    best
}

fn leading_whitespace_bytes(input: &str) -> usize {
    input
        .char_indices()
        .take_while(|(_, character)| character.is_whitespace())
        .map(|(index, character)| index + character.len_utf8())
        .last()
        .unwrap_or(0)
}

const fn is_json_whitespace(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\r' | '\n')
}

fn leading_json_whitespace_bytes(input: &str) -> usize {
    input
        .char_indices()
        .take_while(|(_, character)| is_json_whitespace(*character))
        .map(|(index, character)| index + character.len_utf8())
        .last()
        .unwrap_or(0)
}

fn push_text(events: &mut Vec<ToolStreamEvent>, fragment: String) {
    if fragment.is_empty() {
        return;
    }
    if let Some(ToolStreamEvent::Text(existing)) = events.last_mut() {
        existing.push_str(&fragment);
    } else {
        events.push(ToolStreamEvent::Text(fragment));
    }
}

fn push_arguments(events: &mut Vec<ToolStreamEvent>, index: usize, fragment: String) {
    if fragment.is_empty() {
        return;
    }
    if let Some(ToolStreamEvent::ArgumentsDelta {
        index: existing_index,
        fragment: existing,
    }) = events.last_mut()
    {
        if *existing_index == index {
            existing.push_str(&fragment);
            return;
        }
    }
    events.push(ToolStreamEvent::ArgumentsDelta { index, fragment });
}

fn escape_json_character(character: char, output: &mut String) {
    match character {
        '"' => output.push_str("\\\""),
        '\\' => output.push_str("\\\\"),
        '\u{08}' => output.push_str("\\b"),
        '\u{0c}' => output.push_str("\\f"),
        '\n' => output.push_str("\\n"),
        '\r' => output.push_str("\\r"),
        '\t' => output.push_str("\\t"),
        value if value <= '\u{1f}' => {
            use std::fmt::Write as _;
            let _ = write!(output, "\\u{:04x}", u32::from(value));
        }
        value => output.push(value),
    }
}

#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]
#[cfg(test)]
mod tests {
    use super::{
        IncrementalToolCallTracker, MAX_JSON_NESTING, ToolParseError, ToolStreamEvent,
        ToolStreamEvent::{ArgumentsDelta, Text, ToolEnd, ToolStart},
    };
    use crate::tool_parser::{MAX_INSIDE_TOOL_CALL_BYTES, ToolSchema, parse_tool_calls};

    fn string_schema(function: &str, params: &[&str]) -> ToolSchema {
        let properties = params
            .iter()
            .map(|name| ((*name).to_owned(), serde_json::json!({ "type": "string" })))
            .collect::<serde_json::Map<String, serde_json::Value>>();
        ToolSchema::from_tools(Some(&[serde_json::json!({
            "type": "function",
            "function": {
                "name": function,
                "parameters": { "type": "object", "properties": properties }
            }
        })]))
        .expect("test schema has typed properties")
    }

    fn schema(function: &str, properties: &serde_json::Value) -> ToolSchema {
        ToolSchema::from_tools(Some(&[serde_json::json!({
            "type": "function",
            "function": {
                "name": function,
                "parameters": { "type": "object", "properties": properties }
            }
        })]))
        .expect("test schema has typed properties")
    }

    fn collect(
        tracker: &mut IncrementalToolCallTracker,
        chunks: &[&str],
    ) -> (Vec<ToolStreamEvent>, Option<ToolParseError>) {
        let mut events = Vec::new();
        for chunk in chunks {
            let output = tracker.process(chunk);
            events.extend(output.events);
            if output.error.is_some() {
                return (events, output.error);
            }
        }
        let output = tracker.finish();
        events.extend(output.events);
        (events, output.error)
    }

    fn arguments(events: &[ToolStreamEvent], index: usize) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                ArgumentsDelta { index: i, fragment } if *i == index => Some(fragment.as_str()),
                Text(_) | ToolStart { .. } | ArgumentsDelta { .. } | ToolEnd { .. } => None,
            })
            .collect()
    }

    fn random_chunks(input: &str, mut seed: u64) -> Vec<&str> {
        let boundaries: Vec<usize> = input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
            .collect();
        let mut chunks = Vec::new();
        let mut cursor = 0;
        while cursor + 1 < boundaries.len() {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let width = usize::try_from((seed >> 32) % 11 + 1).unwrap_or(1);
            let end = (cursor + width).min(boundaries.len() - 1);
            chunks.push(&input[boundaries[cursor]..boundaries[end]]);
            cursor = end;
        }
        chunks
    }

    #[test]
    fn qwen_emits_start_and_string_progress_before_closers() {
        let mut tracker = IncrementalToolCallTracker::new(
            true,
            Some(string_schema("write_file", &["path", "content"])),
        );

        let start = tracker.process(
            "intro <tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/a\n</parameter>\n<parameter=content>\nhello ",
        );

        assert_eq!(
            start.events.first(),
            Some(&Text("intro ".to_owned())),
            "text preceding a call keeps its original order"
        );
        assert!(start.events.contains(&ToolStart {
            index: 0,
            name: "write_file".to_owned(),
        }));
        assert!(arguments(&start.events, 0).contains("hello"));
        assert!(start.error.is_none());
        assert!(tracker.holding());
        assert_eq!(tracker.completed_call_count(), 0);

        let end = tracker.process("world\n</parameter>\n</function>\n</tool_call>");
        assert!(end.events.contains(&ToolEnd { index: 0 }));
        assert!(end.error.is_none());
        assert_eq!(tracker.completed_call_count(), 1);
        assert!(!tracker.holding());

        let mut all = start.events;
        all.extend(end.events);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments(&all, 0)).unwrap(),
            serde_json::json!({"path": "/tmp/a", "content": "hello world"})
        );
    }

    #[test]
    fn qwen_escapes_declared_strings_incrementally() {
        let mut tracker =
            IncrementalToolCallTracker::new(true, Some(string_schema("write", &["content"])));
        let input = "<tool_call>\n<function=write>\n<parameter=content>\nquote: \" slash: \\ line\nCafé </parame";
        let first = tracker.process(input);
        assert!(arguments(&first.events, 0).contains(r#"quote: \" slash: \\"#));
        assert!(first.error.is_none());

        let second = tracker.process("ter>\n</function>\n</tool_call>");
        let mut events = first.events;
        events.extend(second.events);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments(&events, 0)).unwrap(),
            serde_json::json!({"content": "quote: \" slash: \\ line\nCafé "})
        );
    }

    #[test]
    fn legacy_json_streams_nested_arguments_without_waiting_for_close_tag() {
        let mut tracker = IncrementalToolCallTracker::new(true, None);
        let first = tracker.process(
            r#"<tool_call>{"name":"search","arguments":{"filters":{"tags":["rust","ml"],"exact":true},"query":"hel"#,
        );
        assert!(first.events.contains(&ToolStart {
            index: 0,
            name: "search".to_owned(),
        }));
        assert!(arguments(&first.events, 0).contains(r#""filters""#));
        assert!(first.error.is_none());

        let second = tracker.process(r#"lo"}}</tool_call>"#);
        let mut events = first.events;
        events.extend(second.events);
        assert!(events.contains(&ToolEnd { index: 0 }));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments(&events, 0)).unwrap(),
            serde_json::json!({
                "filters": {"tags": ["rust", "ml"], "exact": true},
                "query": "hello"
            })
        );
    }

    #[test]
    fn legacy_arguments_before_name_are_released_after_identity() {
        let mut tracker = IncrementalToolCallTracker::new(true, None);
        let held = tracker.process(r#"<tool_call>{"arguments":{"x":[1,2,3]},"na"#);
        assert!(held.events.is_empty());
        assert!(held.error.is_none());
        assert!(tracker.holding());

        let released = tracker.process(r#"me":"later"}</tool_call>"#);
        assert_eq!(
            released.events.first(),
            Some(&ToolStart {
                index: 0,
                name: "later".to_owned()
            })
        );
        assert_eq!(arguments(&released.events, 0), r#"{"x":[1,2,3]}"#);
        assert_eq!(released.events.last(), Some(&ToolEnd { index: 0 }));
    }

    #[test]
    fn minicpm_cdata_string_streams_and_ignores_function_text_inside_cdata() {
        let mut tracker =
            IncrementalToolCallTracker::new(true, Some(string_schema("write", &["code"])));
        let first = tracker.process(
            "<function name=\"write\"><param name=\"code\"><![CDATA[fn main() { // </function>",
        );
        assert!(first.events.contains(&ToolStart {
            index: 0,
            name: "write".to_owned(),
        }));
        assert!(arguments(&first.events, 0).contains("fn main"));
        assert!(!first.events.contains(&ToolEnd { index: 0 }));

        let second = tracker.process("\n}]]></param></function>");
        let mut events = first.events;
        events.extend(second.events);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments(&events, 0)).unwrap(),
            serde_json::json!({"code": "fn main() { // </function>\n}"})
        );
        assert_eq!(events.last(), Some(&ToolEnd { index: 0 }));
    }

    #[test]
    fn declared_object_streams_then_reports_malformed_type_without_replay() {
        let typed = schema(
            "configure",
            &serde_json::json!({"config": {"type": "object"}}),
        );
        let mut tracker = IncrementalToolCallTracker::new(true, Some(typed));
        let first =
            tracker.process("<tool_call><function=configure><parameter=config>{\"nested\":[1,2]");
        assert!(arguments(&first.events, 0).contains("nested"));
        assert!(first.error.is_none());

        let failed = tracker.process("oops</parameter></function></tool_call>");
        assert!(matches!(
            failed.error,
            Some(ToolParseError::InvalidArgumentType { .. } | ToolParseError::Malformed { .. })
        ));
        assert!(failed.events.iter().all(|event| !matches!(event, Text(_))));
        assert!(!failed.events.contains(&ToolEnd { index: 0 }));
    }

    #[test]
    fn duplicate_arguments_are_rejected_before_tool_end() {
        let mut tracker = IncrementalToolCallTracker::new(true, None);
        let output =
            tracker.process(r#"<tool_call>{"name":"dup","arguments":{"x":1,"x":2}}</tool_call>"#);
        assert!(matches!(
            output.error,
            Some(ToolParseError::DuplicateArgument { ref name }) if name == "x"
        ));
        assert!(!output.events.contains(&ToolEnd { index: 0 }));
    }

    #[test]
    fn duplicate_xml_parameter_is_rejected() {
        let mut tracker = IncrementalToolCallTracker::new(true, Some(string_schema("dup", &["x"])));
        let output = tracker.process(
            "<tool_call><function=dup><parameter=x>a</parameter><parameter=x>b</parameter></function></tool_call>",
        );
        assert!(matches!(
            output.error,
            Some(ToolParseError::DuplicateArgument { ref name }) if name == "x"
        ));
        assert!(!output.events.contains(&ToolEnd { index: 0 }));
    }

    #[test]
    fn schema_absence_holds_ambiguous_xml_value_until_close() {
        let mut tracker = IncrementalToolCallTracker::new(true, None);
        let first = tracker.process("<tool_call><function=f><parameter=value>[1,2");
        assert!(first.events.contains(&ToolStart {
            index: 0,
            name: "f".to_owned()
        }));
        assert_eq!(arguments(&first.events, 0), r#"{"value":"#);

        let second = tracker.process("]</parameter></function></tool_call>");
        let mut events = first.events;
        events.extend(second.events);
        assert_eq!(arguments(&events, 0), r#"{"value":[1,2]}"#);
        assert_eq!(events.last(), Some(&ToolEnd { index: 0 }));
    }

    #[test]
    fn malformed_declared_integer_is_a_typed_terminal_error() {
        let typed = schema("set", &serde_json::json!({"count": {"type": "integer"}}));
        let mut tracker = IncrementalToolCallTracker::new(true, Some(typed));
        let output = tracker.process(
            "<tool_call><function=set><parameter=count>3.14</parameter></function></tool_call>",
        );
        assert!(matches!(
            output.error,
            Some(ToolParseError::InvalidArgumentType {
                ref name,
                expected: "integer"
            }) if name == "count"
        ));
        assert!(!output.events.contains(&ToolEnd { index: 0 }));
    }

    #[test]
    fn declared_object_does_not_treat_xml_closer_inside_json_string_as_markup() {
        let typed = schema("set", &serde_json::json!({"config": {"type": "object"}}));
        let mut tracker = IncrementalToolCallTracker::new(true, Some(typed));
        let (events, error) = collect(
            &mut tracker,
            &[
                "<tool_call><function=set><parameter=config>{\"text\":\"</para",
                "meter> is data\"}</parameter></function></tool_call>",
            ],
        );
        assert!(error.is_none());
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments(&events, 0)).unwrap(),
            serde_json::json!({"config": {"text": "</parameter> is data"}})
        );
        assert_eq!(events.last(), Some(&ToolEnd { index: 0 }));
    }

    #[test]
    fn declared_number_accepts_trailing_xml_wrapper_whitespace() {
        let typed = schema("set", &serde_json::json!({"count": {"type": "number"}}));
        let mut tracker = IncrementalToolCallTracker::new(true, Some(typed));
        let (events, error) = collect(
            &mut tracker,
            &[
                "<tool_call><function=set><parameter=count>\n42 \n</parameter></function></tool_call>",
            ],
        );
        assert!(error.is_none());
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments(&events, 0)).unwrap(),
            serde_json::json!({"count": 42})
        );
    }

    #[test]
    fn legacy_scalar_arguments_complete_before_envelope_delimiter() {
        let mut tracker = IncrementalToolCallTracker::new(true, None);
        let (events, error) = collect(
            &mut tracker,
            &[r#"<tool_call>{"name":"scalar","arguments":42}</tool_call>"#],
        );
        assert!(error.is_none());
        assert_eq!(arguments(&events, 0), "42");
        assert_eq!(events.last(), Some(&ToolEnd { index: 0 }));
    }

    #[test]
    fn malformed_json_trailing_comma_is_rejected() {
        let mut tracker = IncrementalToolCallTracker::new(true, None);
        let output =
            tracker.process(r#"<tool_call>{"name":"bad","arguments":{"x":1,}}</tool_call>"#);
        assert!(matches!(
            output.error,
            Some(ToolParseError::Malformed { .. })
        ));
        assert!(!output.events.contains(&ToolEnd { index: 0 }));
    }

    #[test]
    fn legacy_envelope_trailing_comma_is_rejected() {
        let mut tracker = IncrementalToolCallTracker::new(true, None);
        let output = tracker.process(r#"<tool_call>{"name":"bad","arguments":{},}</tool_call>"#);
        assert!(matches!(
            output.error,
            Some(ToolParseError::Malformed { .. })
        ));
        assert!(!output.events.contains(&ToolEnd { index: 0 }));
    }

    #[test]
    fn legacy_rejects_unicode_whitespace_outside_json_strings() {
        let mut tracker = IncrementalToolCallTracker::new(true, None);
        let output =
            tracker.process("<tool_call>{\u{a0}\"name\":\"bad\",\"arguments\":{}}</tool_call>");
        assert!(matches!(
            output.error,
            Some(ToolParseError::Malformed { .. })
        ));
        assert!(!output.events.contains(&ToolEnd { index: 0 }));
    }

    #[test]
    fn declared_number_rejects_unicode_json_whitespace() {
        let typed = schema("set", &serde_json::json!({"count": {"type": "number"}}));
        let mut tracker = IncrementalToolCallTracker::new(true, Some(typed));
        let output = tracker.process(
            "<tool_call><function=set><parameter=count>42\u{a0}</parameter></function></tool_call>",
        );
        assert!(matches!(
            output.error,
            Some(ToolParseError::Malformed { .. })
        ));
        assert!(!output.events.contains(&ToolEnd { index: 0 }));
    }

    #[test]
    fn multiple_calls_keep_monotonic_indices_and_event_order() {
        let input = concat!(
            "before",
            r#"<tool_call>{"name":"one","arguments":{}}</tool_call>"#,
            "middle",
            r#"<function name="two"><param name="x">true</param></function>"#,
            "after"
        );
        let mut tracker = IncrementalToolCallTracker::new(true, None);
        let (events, error) = collect(&mut tracker, &[input]);
        assert!(error.is_none());
        assert_eq!(
            events
                .iter()
                .filter_map(|event| match event {
                    Text(text) => Some(text.as_str()),
                    ToolStart { .. } | ArgumentsDelta { .. } | ToolEnd { .. } => None,
                })
                .collect::<String>(),
            "beforemiddleafter"
        );
        let starts: Vec<(usize, &str)> = events
            .iter()
            .filter_map(|event| match event {
                ToolStart { index, name } => Some((*index, name.as_str())),
                Text(_) | ArgumentsDelta { .. } | ToolEnd { .. } => None,
            })
            .collect();
        assert_eq!(starts, vec![(0, "one"), (1, "two")]);
        assert_eq!(tracker.completed_call_count(), 2);
    }

    #[test]
    fn finish_reports_an_exposed_incomplete_call_without_prose_replay() {
        let mut tracker =
            IncrementalToolCallTracker::new(true, Some(string_schema("write", &["content"])));
        let first = tracker.process("<tool_call><function=write><parameter=content>partial data");
        assert!(
            first
                .events
                .iter()
                .any(|event| matches!(event, ToolStart { .. }))
        );

        let finish = tracker.finish();
        assert_eq!(finish.error, Some(ToolParseError::IncompleteToolCall));
        assert!(finish.events.is_empty());
    }

    #[test]
    fn inactive_parser_is_ordered_text_passthrough() {
        let mut tracker = IncrementalToolCallTracker::new(false, None);
        let (events, error) = collect(
            &mut tracker,
            &["hello", "<tool_call>{\"name\":\"x\"}</tool_call>"],
        );
        assert_eq!(
            events,
            vec![
                Text("hello".to_owned()),
                Text("<tool_call>{\"name\":\"x\"}</tool_call>".to_owned())
            ]
        );
        assert!(error.is_none());
    }

    #[test]
    fn valid_fixtures_match_batch_at_every_char_boundary() {
        let fixtures = [
            r#"before <tool_call>{"name":"json","arguments":{"s":"a } </tool_cal b","n":[1,{"x":true}]}}</tool_call> after"#,
            "<tool_call>\n<function=xml>\n<parameter=value>\nhello\nworld\n</parameter>\n</function>\n</tool_call>",
            r#"<function name="mini"><param name="value"><![CDATA[hello </function> world]]></param></function>"#,
        ];

        for fixture in fixtures {
            let expected = parse_tool_calls(fixture, None);
            let boundaries: Vec<usize> = fixture
                .char_indices()
                .map(|(index, _)| index)
                .chain(std::iter::once(fixture.len()))
                .collect();
            let chunks: Vec<&str> = boundaries
                .windows(2)
                .map(|window| &fixture[window[0]..window[1]])
                .collect();
            let mut tracker = IncrementalToolCallTracker::new(true, None);
            let (events, error) = collect(&mut tracker, &chunks);
            assert!(error.is_none(), "fixture failed: {fixture:?}: {error:?}");
            let text: String = events
                .iter()
                .filter_map(|event| match event {
                    Text(fragment) => Some(fragment.as_str()),
                    ToolStart { .. } | ArgumentsDelta { .. } | ToolEnd { .. } => None,
                })
                .collect::<String>()
                .trim()
                .to_owned();
            assert_eq!(text, expected.text);
            assert_eq!(tracker.completed_call_count(), expected.tool_calls.len());
            for (index, expected_call) in expected.tool_calls.iter().enumerate() {
                assert!(events.contains(&ToolStart {
                    index,
                    name: expected_call.name.clone()
                }));
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&arguments(&events, index)).unwrap(),
                    expected_call.arguments
                );
                assert!(events.contains(&ToolEnd { index }));
            }
        }
    }

    #[test]
    fn legacy_close_tag_inside_json_string_is_data() {
        let input =
            r#"<tool_call>{"name":"json","arguments":{"s":"a </tool_call> b"}}</tool_call>"#;
        let chunks: Vec<&str> = input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
            .collect::<Vec<_>>()
            .windows(2)
            .map(|window| &input[window[0]..window[1]])
            .collect();
        let mut tracker = IncrementalToolCallTracker::new(true, None);
        let (events, error) = collect(&mut tracker, &chunks);
        assert!(error.is_none());
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments(&events, 0)).unwrap(),
            serde_json::json!({"s": "a </tool_call> b"})
        );
        assert_eq!(events.last(), Some(&ToolEnd { index: 0 }));
    }

    #[test]
    fn randomized_unicode_qwen_boundaries_reconstruct_exactly() {
        let input = "prefix <tool_call>\n<function=write>\n<parameter=content>\nCafé 🚀 \\\"\nline </function>\n</parameter>\n</function>\n</tool_call> suffix";
        for seed in 0..32 {
            let mut tracker =
                IncrementalToolCallTracker::new(true, Some(string_schema("write", &["content"])));
            let chunks = random_chunks(input, seed);
            let (events, error) = collect(&mut tracker, &chunks);
            assert!(error.is_none(), "seed {seed} failed: {error:?}");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&arguments(&events, 0)).unwrap(),
                serde_json::json!({"content": "Café 🚀 \\\"\nline </function>"}),
                "seed {seed} reconstructed different arguments"
            );
            assert_eq!(tracker.completed_call_count(), 1);
        }
    }

    #[test]
    fn declared_large_array_streams_progress_with_bounded_parser_retention() {
        let typed = schema("store", &serde_json::json!({"items": {"type": "array"}}));
        let numbers = (0..20_000)
            .map(|number| number.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let first_chunk = format!("<tool_call><function=store><parameter=items>[{numbers}");
        let mut tracker = IncrementalToolCallTracker::new(true, Some(typed));
        let first = tracker.process(&first_chunk);
        assert!(first.error.is_none());
        assert!(arguments(&first.events, 0).len() >= numbers.len());
        assert!(!first.events.contains(&ToolEnd { index: 0 }));
        assert!(
            tracker.buffer.len() < 64,
            "consumed structured payload must not remain in the input buffer"
        );

        let second = tracker.process("]</parameter></function></tool_call>");
        let mut events = first.events;
        events.extend(second.events);
        let parsed: serde_json::Value = serde_json::from_str(&arguments(&events, 0)).unwrap();
        assert_eq!(parsed["items"].as_array().unwrap().len(), 20_000);
        assert_eq!(events.last(), Some(&ToolEnd { index: 0 }));
    }

    #[test]
    fn excessive_json_nesting_fails_at_a_bounded_depth() {
        let mut tracker = IncrementalToolCallTracker::new(true, None);
        let input = format!(
            "<tool_call>{{\"name\":\"deep\",\"arguments\":{}}}",
            "[".repeat(MAX_JSON_NESTING + 1)
        );
        let output = tracker.process(&input);
        assert_eq!(output.error, Some(ToolParseError::NestingTooDeep));
        assert!(!output.events.contains(&ToolEnd { index: 0 }));
    }

    #[test]
    fn call_limit_is_enforced_after_partial_argument_events() {
        let mut tracker =
            IncrementalToolCallTracker::new(true, Some(string_schema("write", &["content"])));
        let opener = "<tool_call><function=write><parameter=content>";
        let first = tracker.process(opener);
        assert!(first.error.is_none());
        let huge = "x".repeat(MAX_INSIDE_TOOL_CALL_BYTES + 1);
        let overflow = tracker.process(&huge);
        assert_eq!(
            overflow.error,
            Some(ToolParseError::CallTooLarge {
                limit: MAX_INSIDE_TOOL_CALL_BYTES
            })
        );
        assert!(
            overflow
                .events
                .iter()
                .all(|event| !matches!(event, Text(_)))
        );
        assert!(!overflow.events.contains(&ToolEnd { index: 0 }));
    }
}
