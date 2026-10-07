use super::*;

#[test]
fn manifest_matches_protocol_12_contract() {
    let manifest = manifest();
    assert_eq!(manifest.name, PLUGIN_NAME);
    assert_eq!(manifest.version, PLUGIN_VERSION);
    assert_eq!(manifest.protocol.as_deref(), Some("1.2"));
    assert!(
        manifest
            .capabilities
            .contains(&PROVIDER_CREDENTIALS.to_string())
    );
    assert!(
        manifest
            .capabilities
            .contains(&gray_plugin::capabilities::HOST_SAY.to_string())
    );
    assert_eq!(manifest.providers.len(), 1);
    let provider = &manifest.providers[0];
    assert_eq!(provider.id, PROVIDER_ID);
    // Relay sidecar: the host must not verbatim-replay its requests, and
    // Claude's native prompt cache lives ~1h — declared so the host's
    // warmth timer and cold-cache notices use the real lifetime.
    let request = &provider.transport.request;
    assert!(!request.warm_replay);
    assert_eq!(request.cache_ttl_secs, Some(3600));
    let method = &provider.auth_methods[0];
    assert_eq!(method.id, AUTH_METHOD_ID);
    assert_eq!(method.operations, vec!["models", "chat"]);
    // Transport placeholder survives validation (rewritten per turn).
    provider
        .validate()
        .expect("placeholder declaration validates");
}
