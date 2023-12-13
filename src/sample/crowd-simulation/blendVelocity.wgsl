@binding(0) @group(0) var<uniform> params : SimParams;
@binding(1) @group(0) var<storage, read> agents_r : Agents;
@binding(2) @group(0) var<storage, read_write> agents_w : Agents;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  var index = GlobalInvocationID.x;
  var agent = agents_r.agents[index];
  var goal = agent.goal;

  // TODO: consider ignoring the particle when it's close enough to the goal

  // 4.1 velocity blending 
  // (https://github.com/wayne-wu/webgpu-crowd-simulation/blob/main/src/shaders/explicitIntegration.compute.wgsl)
  var vp = normalize(goal - agent.pos) * agentSpeed;
  agent.vel = (1.0 - blendFactor) * agent.vel + blendFactor * vp;
  agent.ppos = agent.pos + params.deltaT * agent.vel;

  // Write back
  agents_w.agents[index] = agent;
}
