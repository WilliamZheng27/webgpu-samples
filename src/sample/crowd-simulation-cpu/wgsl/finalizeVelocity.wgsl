@binding(0) @group(0) var<uniform> params : SimParams;
@binding(1) @group(0) var<storage, read> agents_r : Agents;
@binding(2) @group(0) var<storage, read_write> agents_w : Agents;

fn poly6Kernel(r : f32) -> f32 {
  var w = 0.0;
  var xsph_c = 217.0;
  if (eps <= r && r <= xsph_c) {
      w = 315.0 / (64.0 * 3.14159 * pow(xsph_c, 9.0));
      var hmr = xsph_c * xsph_c - r * r;
      w = w * hmr * hmr * hmr;
  }
  return w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  var index = GlobalInvocationID.x;
  var agent = agents_r.agents[index];

  agent.vel = (agent.ppos - agent.pos) / params.deltaT;

  // 4.3 Cohesion (adding XSPH viscosity)
  var avgVel = vec2<f32>(0.0);

  for (var j = 0u; j < arrayLength(&agents_r.agents); j++) {
    if (index == j) {
      continue;
    }

    var agent_j = agents_r.agents[j];
    var equality = (agent_j.goal == agent.goal);
    if (!equality.x || !equality.y) {
      continue;  
    }

    var d = distance(agent.ppos, agent_j.ppos);
    if (d > cohesionRadius * params.agentScale){
      continue;
    }
    var w = poly6Kernel(d*d);
    avgVel = avgVel + (agent.vel - agent_j.vel) * w;
  }
  var xsph_h = 7.0;
  agent.vel = agent.vel + xsph_h * avgVel;

  // 4.6 Maximum Speed and Acceleration Limiting
  var dir = normalize(agent.vel);
  var maxSpeed : f32 = 1.2 * agentSpeed * params.agentScale;
  if(length(agent.vel) > maxSpeed){
    agent.vel = maxSpeed * dir;
  }

  // finally, update position
  agent.pos = agent.pos + agent.vel * params.deltaT;

  // agent.dir = dir_blending * normalize(agent.dir) + (1.0 - dir_blending) * v_dir;

  // Write back
  agents_w.agents[index] = agent;
}