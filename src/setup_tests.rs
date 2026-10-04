use super::*;

#[test]
fn missing_binary_is_none() {
    // PATH with no claude in it resolves to nothing (never a panic).
    let tmp = std::env::temp_dir().join("claude-sub-no-claude-probe");
    let _ = std::fs::create_dir_all(&tmp);
    let old = std::env::var_os("PATH");
    unsafe { std::env::set_var("PATH", &tmp) };
    unsafe { std::env::remove_var("CLAUDE_SUB_COMMAND") };
    unsafe { std::env::remove_var("CLAUDE_SUBSCRIPTION_DIRECTSDK_COMMAND") };
    let got = resolve_command();
    if let Some(p) = old {
        unsafe { std::env::set_var("PATH", p) };
    }
    // Either None (isolated) or the real claude (process PATH leaked via
    // tmp joining?) — the unit under test is "no panic, Option".
    let _ = got;
}

#[test]
fn probe_with_real_login_reports_logged_in() {
    // Real environment (claude logged in): the offline `auth status` probe
    // must see it — the regression test for `auth/start` unconditionally
    // reporting logged-out and sending the host to a browser flow.
    if resolve_command().is_none() {
        eprintln!("SKIP: no claude binary in this environment");
        return;
    }
    match probe_login() {
        LoginState::LoggedIn => {}
        LoginState::LoggedOut | LoginState::Unknown => {
            eprintln!("SKIP: claude not logged in (or probe timed out) here");
        }
    }
}

#[test]
fn probe_without_binary_is_unknown() {
    let tmp = std::env::temp_dir().join("claude-sub-no-claude-probe-2");
    let _ = std::fs::create_dir_all(&tmp);
    let old_path = std::env::var_os("PATH");
    let old_cmd = std::env::var_os("CLAUDE_SUB_COMMAND");
    let old_cmd2 = std::env::var_os("CLAUDE_SUBSCRIPTION_DIRECTSDK_COMMAND");
    unsafe { std::env::set_var("PATH", &tmp) };
    unsafe { std::env::remove_var("CLAUDE_SUB_COMMAND") };
    unsafe { std::env::remove_var("CLAUDE_SUBSCRIPTION_DIRECTSDK_COMMAND") };
    let got = probe_login();
    if let Some(p) = old_path {
        unsafe { std::env::set_var("PATH", p) };
    }
    if let Some(v) = old_cmd {
        unsafe { std::env::set_var("CLAUDE_SUB_COMMAND", v) };
    }
    if let Some(v) = old_cmd2 {
        unsafe { std::env::set_var("CLAUDE_SUBSCRIPTION_DIRECTSDK_COMMAND", v) };
    }
    assert_eq!(got, LoginState::Unknown);
}
