use cloth_simulation_lib::{model, run};

mod cloth_generator;

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
    match args.len() {
        0 => Err(ProgramError::InvalidUsage(
            "Missing all arguments. How did you do that?".to_string(),
        )),
        1 => Err(ProgramError::InvalidUsage(format!(
            "Usage: {} [<config-file> | --cloth <height> <width>]",
            args[0]
        ))),
        2 => {
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
        4 => {
            if args[1] == "--cloth" {
                let height = args[2].parse::<u32>().unwrap();
                let width = args[3].parse::<u32>().unwrap();

                let mesh = cloth_generator::generate_cloth_mesh(height, width);

                pollster::block_on(run(mesh));
                Ok(())
            } else {
                Err(ProgramError::InvalidUsage(format!(
                    "Usage: {} [<config-file> | --cloth <height> <width>]",
                    args[0]
                )))
            }
        }
        _ => Err(ProgramError::InvalidUsage("Too many arguments".to_string())),
    }
}
