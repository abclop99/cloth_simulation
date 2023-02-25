//use serde_json::Result;

use cloth_simulation_lib::{model, run};

#[derive(Debug)]
enum ProgramError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidUsage(String),
}

impl From<std::io::Error> for ProgramError {
    fn from(error: std::io::Error) -> Self {
        ProgramError::Io(error)
    }
}

fn main() -> Result<(), ProgramError> {
    let args: Vec<String> = std::env::args().collect();

    // Check if the user provided a config file for the cloth simulation
    if args.len() != 2 {
        if args.is_empty() {
            Err(ProgramError::InvalidUsage(
                "Missing all arguments".to_string(),
            ))
        } else {
            Err(ProgramError::InvalidUsage(
                "Missing simulation config file".to_string(),
            ))
        }
    } else {
        let contents = std::fs::read_to_string(&args[1])?;

        let mesh: serde_json::Result<model::Mesh> = serde_json::from_str(&contents);

        match mesh {
            Err(e) => Err(ProgramError::Json(e)),
            Ok(mesh) => {
                pollster::block_on(run(mesh));
                Ok(())
            }
        }
    }
}
