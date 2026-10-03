//! Library target for `gray-claude-sub`.
//!
//! Provides both:
//! 1. The protocol-1.2 sidecar plugin (`claude-sub` binary) for Gray's plugin system.
//! 2. The direct in-process `Provider` implementation (`direct_provider`) implementing `gray_core::agent::Provider`.

pub mod catalog;
pub mod chat;
pub mod direct_provider;
pub mod manifest;
pub mod models;
pub mod relay;
pub mod setup;
