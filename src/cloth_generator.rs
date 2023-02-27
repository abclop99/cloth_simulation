use cloth_simulation_lib::model;

const GRAVITY_AMOUNT: f32 = -3.0;

const VERTEX_COLOR: [f32; 3] = [0.6, 0.4, 0.2];
const VERTEX_MASS: f32 = 0.01;

const SPRING_CONSTANT: f32 = 100.0;
const SPRING_DAMPING: f32 = 0.5;

const BENDING_SPRING_CONSTANT: f32 = 50.0;
const BENDING_SPRING_DAMPING: f32 = 0.3;

struct Position {
    row: u32,
    column: u32,
}

pub fn generate_cloth_mesh(height: u32, width: u32) -> model::Mesh {
    let name = format!("Cloth: {}x{}", height, width);
    let settings = model::SimulationSettings {
        gravity: [0.0, GRAVITY_AMOUNT, 0.0],
        wind: [0.2, 0.0, 0.05],
        fluid_density: 1.2,
        ground_level: -1.0,
        ground_size: 2.0,
        ground_friction_static: 0.5,
        ground_friction_dynamic: 0.3,
        ground_restitution: 0.05,
        ground_color: [0.1, 0.5, 0.05],
    };
    let vertices = generate_vertices(height, width);
    let springs = generate_springs(height, width);
    let triangles = generate_triangles(height, width);

    model::Mesh {
        name,
        settings,
        vertices,
        springs,
        triangles,
    }
}

fn generate_vertices(height: u32, width: u32) -> Vec<model::Vertex> {
    let mut vertices = Vec::new();

    for row in 0..height {
        for col in 0..width {
            // Normalize position so the width is 1 and the position is
            // centered horizontally.
            let normalized_x = -(col as f32 / (width - 1) as f32) + 0.5;
            let normalized_z = -(row as f32 / (width - 1) as f32) + 1.0;

            let position = [normalized_x, normalized_z, 0.0];
            let color = VERTEX_COLOR;
            let normal = [0.0; 3]; // Will be calculated later
            let mass = VERTEX_MASS;
            let fixed = if row == 0 { 1 } else { 0 };
            let velocity = [0.0; 3];

            vertices.push(model::Vertex {
                position,
                color,
                normal,
                mass,
                fixed,
                velocity,
            });
        }
    }

    vertices
}

fn generate_springs(height: u32, width: u32) -> Vec<model::Spring> {
    let mut springs = Vec::new();

    // Generate springs for each square in the grid
    for col in 0..width {
        for row in 0..height - 1 {
            // Vertices for each small square
            let top_left = Position { row, column: col };
            let top_right = Position {
                row,
                column: col + 1,
            };
            let bottom_left = Position {
                row: row + 1,
                column: col,
            };
            let bottom_right = Position {
                row: row + 1,
                column: col + 1,
            };
            let top_left_index = top_left.to_index(width);
            let top_right_index = top_right.to_index(width);
            let bottom_left_index = bottom_left.to_index(width);
            let bottom_right_index = bottom_right.to_index(width);

            // Vertices for each large square for bending springs
            let top_left_bend = Position { row, column: col };
            let top_right_bend = Position {
                row,
                column: col + 2,
            };
            let bottom_left_bend = Position {
                row: row + 2,
                column: col,
            };
            let bottom_right_bend = Position {
                row: row + 2,
                column: col + 2,
            };
            let top_left_bend_index = top_left_bend.to_index(width);
            let top_right_bend_index = top_right_bend.to_index(width);
            let bottom_left_bend_index = bottom_left_bend.to_index(width);
            let bottom_right_bend_index = bottom_right_bend.to_index(width);

            // Horizontal springs for each square if not on the right edge
            if col < width - 1 {
                springs.push(model::Spring {
                    vertices: [top_left_index, top_right_index],
                    k_s: SPRING_CONSTANT,
                    k_d: SPRING_DAMPING,
                    rest_length: 0.0, // Will be calculated later
                });

                // Bending springs, longer so more limits
                if col < width - 2 {
                    springs.push(model::Spring {
                        vertices: [top_left_bend_index, top_right_bend_index],
                        k_s: BENDING_SPRING_CONSTANT,
                        k_d: BENDING_SPRING_DAMPING,
                        rest_length: 0.0, // Will be calculated later
                    });
                }
            }

            // Vertical springs for each square if not on the bottom edge
            if row < height - 1 {
                springs.push(model::Spring {
                    vertices: [top_left_index, bottom_left_index],
                    k_s: SPRING_CONSTANT,
                    k_d: SPRING_DAMPING,
                    rest_length: 0.0, // Will be calculated later
                });

                // Bending springs, longer so more limits
                if row < height - 2 {
                    springs.push(model::Spring {
                        vertices: [top_left_bend_index, bottom_left_bend_index],
                        k_s: BENDING_SPRING_CONSTANT,
                        k_d: BENDING_SPRING_DAMPING,
                        rest_length: 0.0, // Will be calculated later
                    });
                }
            }

            // Diagonal springs for each square if not on the right or bottom edge
            if col < width - 1 && row < height - 1 {
                springs.push(model::Spring {
                    vertices: [top_left_index, bottom_right_index],
                    k_s: SPRING_CONSTANT,
                    k_d: SPRING_DAMPING,
                    rest_length: 0.0, // Will be calculated later
                });
                springs.push(model::Spring {
                    vertices: [top_right_index, bottom_left_index],
                    k_s: SPRING_CONSTANT,
                    k_d: SPRING_DAMPING,
                    rest_length: 0.0, // Will be calculated later
                });

                // Bending springs, longer so more limits
                if col < width - 2 && row < height - 2 {
                    springs.push(model::Spring {
                        vertices: [top_left_bend_index, bottom_right_bend_index],
                        k_s: BENDING_SPRING_CONSTANT,
                        k_d: BENDING_SPRING_DAMPING,
                        rest_length: 0.0, // Will be calculated later
                    });
                    springs.push(model::Spring {
                        vertices: [top_right_bend_index, bottom_left_bend_index],
                        k_s: BENDING_SPRING_CONSTANT,
                        k_d: BENDING_SPRING_DAMPING,
                        rest_length: 0.0, // Will be calculated later
                    });
                }
            }
        }
    }

    springs
}

fn generate_triangles(height: u32, width: u32) -> Vec<model::Triangle> {
    let mut triangles = Vec::new();

    for col in 0..width - 1 {
        for row in 0..height - 1 {
            let top_left = Position { row, column: col };
            let top_right = Position {
                row,
                column: col + 1,
            };
            let bottom_left = Position {
                row: row + 1,
                column: col,
            };
            let bottom_right = Position {
                row: row + 1,
                column: col + 1,
            };

            let top_left_index = top_left.to_index(width);
            let top_right_index = top_right.to_index(width);
            let bottom_left_index = bottom_left.to_index(width);
            let bottom_right_index = bottom_right.to_index(width);

            triangles.push(model::Triangle(
                top_left_index,
                top_right_index,
                bottom_left_index,
            ));
            triangles.push(model::Triangle(
                top_right_index,
                bottom_right_index,
                bottom_left_index,
            ));
        }
    }

    triangles
}

impl Position {
    fn to_index(&self, width: u32) -> u16 {
        (self.row * width + self.column) as u16
    }
}
