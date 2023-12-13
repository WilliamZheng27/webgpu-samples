@binding(0) @group(0) var<uniform> params : SimParams;
@binding(1) @group(0) var<storage, read> agents_r : Agents;
@binding(2) @group(0) var<storage, read_write> agents_w : Agents;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  var index = GlobalInvocationID.x;
  var agent = agents_r.agents[index];

  var totalDx = vec2<f32>(0.0);
  var neighborCount = 0;
  for (var j = 0u; j < arrayLength(&agents_r.agents); j++) {
    if (index == j) {
      continue;
    }

    var agent_j = agents_r.agents[j];
    var n = agent.ppos - agent_j.ppos;
    var d = length(n);
    if (d > nearRadius) {
      continue;
    }

    var f = d - (2.0 * agentRadius); // assume all agents have same size
    if (f < 0.0) {
      // 4.2 Short Range Collision
      n = normalize(n);
      var w = 0.5; // assume all agents have same weight
      var dx = -w * k_shortrange * f * n;

      // 4.2 Friction (not implemented yet)

      totalDx = totalDx + dx;
      neighborCount = neighborCount + 1;
    }
  }

  if (neighborCount > 0) {
    totalDx = avgCoeff * totalDx / f32(neighborCount);

    agent.pos = agent.pos + totalDx;
    agent.ppos = agent.ppos + totalDx;
  }

  // Write back
  agents_w.agents[index] = agent;
}