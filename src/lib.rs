//! Library target for `gray-claude-sub`: the protocol-1.2 sidecar plugin
//! (`claude-sub` binary) for Gray's plugin system.

pub mod catalog;
pub mod chat;
pub mod keepalive;
pub mod manifest;
pub mod models;
pub mod relay;
pub mod session;
pub mod setup;
