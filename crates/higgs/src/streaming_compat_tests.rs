//! Real HTTP/SSE regression fixtures, compiled only into the test binary.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

fn app(model: &str) -> axum::Router {
    crate::build_router(
        crate::state::test_state_with_stub_engine(model),
        300.0,
        None,
        0,
        1024 * 1024,
        None,
    )
}

fn request(model: &str, anthropic: bool) -> axum::http::Request<axum::body::Body> {
    let tool = json!({"name":"write", "description":"Inert test tool", "parameters": {
        "type":"object", "properties":{"content":{"type":"string"}}, "required":["content"]
    }});
    let tools = if anthropic {
        json!([{"name":"write", "description":"Inert test tool", "input_schema":tool["parameters"]}])
    } else {
        json!([{"type":"function", "function":tool}])
    };
    axum::http::Request::builder()
        .method("POST")
        .uri(if anthropic {
            "/v1/messages"
        } else {
            "/v1/chat/completions"
        })
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            json!({
                "model":model, "messages":[{"role":"user", "content":"write the fixture"}],
                "stream":true, "max_tokens":4096, "tools":tools, "enable_thinking":false
            })
            .to_string(),
        ))
        .unwrap()
}

async fn body(model: &str, anthropic: bool) -> String {
    let response = app(model).oneshot(request(model, anthropic)).await.unwrap();
    assert_eq!(response.status(), 200);
    String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap()
}

fn data_events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

async fn first_argument(body: &mut axum::body::Body, anthropic: bool) -> String {
    let mut buffered = String::new();
    loop {
        let frame = body
            .frame()
            .await
            .expect("stream ended before arguments")
            .unwrap();
        if let Some(bytes) = frame.data_ref() {
            buffered.push_str(std::str::from_utf8(bytes).unwrap());
            if data_events(&buffered).iter().any(|event| {
                if anthropic {
                    event["delta"]["type"] == "input_json_delta"
                } else {
                    event["choices"][0]["delta"]["tool_calls"]
                        .as_array()
                        .is_some_and(|calls| {
                            calls.iter().any(|call| {
                                call["function"]["arguments"]
                                    .as_str()
                                    .is_some_and(|args| !args.is_empty())
                            })
                        })
                }
            }) {
                return buffered;
            }
        }
    }
}

#[tokio::test]
async fn long_call_exposes_arguments_before_generation_finishes() {
    for anthropic in [false, true] {
        let model = "required-stream-script-long";
        let response = app(model).oneshot(request(model, anthropic)).await.unwrap();
        assert_eq!(response.status(), 200);
        let mut body = response.into_body();
        let prefix = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            first_argument(&mut body, anthropic),
        )
        .await
        .expect("server buffered a long call until its closer");
        assert!(!prefix.contains("\"stop_reason\":\"tool_use\""));
        assert!(!prefix.contains("\"finish_reason\":\"tool_calls\""));
        // Body drop must release the worker. The next iteration shares the
        // engine's global GPU gate and would time out if work were orphaned.
        drop(body);
    }
}

#[tokio::test]
async fn interrupted_call_does_not_have_successful_finish() {
    for model in [
        "required-stream-script-malformed",
        "required-stream-script-capacity",
    ] {
        for anthropic in [false, true] {
            let wire = body(model, anthropic).await;
            assert!(!wire.contains("\"finish_reason\":\"tool_calls\""), "{wire}");
            assert!(!wire.contains("\"stop_reason\":\"tool_use\""), "{wire}");
            assert!(wire.contains("error"), "{wire}");
        }
    }
}

#[tokio::test]
async fn disconnected_receiver_prevents_worker_dispatch() {
    let result = tokio::task::spawn_blocking(|| {
        let engine = crate::state::Engine::test_stub("closed-before-worker");
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        drop(receiver);
        engine.generate_streaming_with_thinking(
            &[],
            1,
            &higgs_models::SamplingParams::default(),
            &[],
            false,
            None,
            &sender,
            false,
            false,
            None,
            None,
            None,
        )
    })
    .await
    .unwrap();
    assert!(
        matches!(result, Err(higgs_engine::error::EngineError::Cancelled)),
        "{result:?}"
    );
}

#[tokio::test]
async fn openai_scripted_xml_reconstructs_arguments() {
    let wire = body("required-stream-script-xml", false).await;
    let mut fragments = String::new();
    let mut ids = Vec::new();
    for event in data_events(&wire) {
        if let Some(calls) = event["choices"][0]["delta"]["tool_calls"].as_array() {
            for call in calls {
                if let Some(id) = call["id"].as_str() {
                    ids.push(id.to_owned());
                }
                if let Some(fragment) = call["function"]["arguments"].as_str() {
                    fragments.push_str(fragment);
                }
            }
        }
    }
    assert_eq!(ids.len(), 1, "{wire}");
    assert_eq!(
        serde_json::from_str::<Value>(&fragments).unwrap(),
        json!({"content":"hello\nworld"})
    );
    assert!(wire.contains("\"finish_reason\":\"tool_calls\""), "{wire}");
}

#[tokio::test]
async fn anthropic_scripted_xml_reconstructs_arguments() {
    let wire = body("required-stream-script-xml", true).await;
    let mut fragments = String::new();
    let mut starts = 0;
    for event in data_events(&wire) {
        if event["content_block"]["type"] == "tool_use" {
            starts += 1;
        }
        if let Some(fragment) = event["delta"]["partial_json"].as_str() {
            fragments.push_str(fragment);
        }
    }
    assert_eq!(starts, 1, "{wire}");
    assert_eq!(
        serde_json::from_str::<Value>(&fragments).unwrap(),
        json!({"content":"hello\nworld"})
    );
    assert!(wire.contains("\"stop_reason\":\"tool_use\""), "{wire}");
}

#[tokio::test]
async fn both_protocols_reconstruct_json_and_minicpm() {
    for model in [
        "required-stream-script-json",
        "required-stream-script-minicpm",
    ] {
        for anthropic in [false, true] {
            let wire = body(model, anthropic).await;
            let mut arguments = String::new();
            for event in data_events(&wire) {
                if anthropic {
                    if let Some(fragment) = event["delta"]["partial_json"].as_str() {
                        arguments.push_str(fragment);
                    }
                } else if let Some(calls) = event["choices"][0]["delta"]["tool_calls"].as_array() {
                    for call in calls {
                        if let Some(fragment) = call["function"]["arguments"].as_str() {
                            arguments.push_str(fragment);
                        }
                    }
                }
            }
            assert_eq!(
                serde_json::from_str::<Value>(&arguments).unwrap(),
                json!({"content":"hello\nworld"}),
                "{model} {anthropic}: {wire}"
            );
        }
    }
}

/// Run manually for SDK tests; this never loads a model or executes a tool.
#[tokio::test]
#[ignore = "local SDK compatibility fixture server"]
async fn serve_tool_streaming_fixture() {
    let config = crate::config::HiggsConfig::default();
    let engines = [
        "long",
        "idle",
        "xml",
        "json",
        "minicpm",
        "malformed",
        "capacity",
    ]
    .into_iter()
    .map(|suffix| {
        let name = format!("required-stream-script-{suffix}");
        let engine = std::sync::Arc::new(crate::state::Engine::test_stub(&name));
        (name, engine)
    })
    .collect();
    let router = crate::router::Router::from_config(&config, engines).unwrap();
    let state = std::sync::Arc::new(crate::state::AppState::new(
        router,
        config,
        reqwest::Client::new(),
        None,
    ));
    let fixture = crate::build_router(state, 300.0, None, 0, 1024 * 1024, None);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:19091")
        .await
        .unwrap();
    axum::serve(
        listener,
        fixture.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await
    .unwrap();
}
