use cloth_simulation::run;

fn main() {
    pollster::block_on(run());
}
