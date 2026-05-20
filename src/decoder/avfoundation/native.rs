#[allow(
    unused_variables,
    non_snake_case,
    dead_code,
    non_camel_case_types,
    non_upper_case_globals
)]
mod bindings {
    include!(concat!(
        env!("OUT_DIR"),
        "/decoder_avfoundation_bindings.rs"
    ));
}

pub use bindings::*;
