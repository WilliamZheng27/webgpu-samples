struct Agent {
  pos : vec2<f32>, // current position
  vel : vec2<f32>,
  ppos: vec2<f32>, // planed/predicted position, updated in different stages
  goal: vec2<f32>, // the agent will try to reach this location
}
struct SimParams {
  deltaT : f32,
}
struct Agents {
  agents : array<Agent>,
}

// constants copied from the paper
const agentSpeed : f32 = 1.4;
const nearRadius : f32 = 2.0;      // threshold for SR collision
const farRadius : f32 = 5.0;       // threshold for LR collision
const cohesionRadius : f32 = 5.0;  // threshold for cohesion
const agentRadius : f32 = 0.03;    // agent size
const blendFactor : f32 = 0.0385;
const k_shortrange : f32 = 1.0;
const k_longrange : f32 = 0.15;
const avgCoeff : f32 = 1.2;
const eps : f32 = 0.0001;
const t0 : f32 = 20.0;
