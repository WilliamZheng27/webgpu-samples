@id(1000) override itr : u32;
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
    var d = distance(agent.pos, agent_j.pos);
    if (d > farRadius * params.agentScale) {
      continue;
    }

    var f = d - (2.0 * agentRadius * params.agentScale); // assume all agents have same size
    if (f < 0.0) {
      // 4.4 Long Range Collision
      var r = 2.0 * agentRadius * params.agentScale;
      var r2 = r * r;
      var dt = params.deltaT;

      var dist = distance(agent.pos, agent_j.pos);
      if (dist < r) {
        r2 = (r - dist) * (r - dist);
      }

      var x_ij = agent.pos - agent_j.pos; // relative displacement
      var v_ij = (1.0/dt) * (agent.ppos - agent.pos - agent_j.ppos + agent_j.pos); // relative velocity

      var a = dot(v_ij, v_ij);
      var b = -dot(x_ij, v_ij);
      var c = dot(x_ij, x_ij) - r2;
      var discr = b*b - a*c;
      if (discr < 0.0 || abs(a) < eps) { continue; }

      discr = sqrt(discr);

      // Compute exact time to collision
      var t = (b - discr)/a;

      // Prune out invalid case
      if (t < eps || t > t0) { continue; }

      // Get time before and after collision
      var t_nocollision = dt * floor(t/dt);
      var t_collision = dt + t_nocollision;

      // Get collision and collision-free positions
      var xi_nocollision = agent.pos + t_nocollision * agent.vel;
      var xi_collision   = agent.pos + t_collision * agent.vel;
      var xj_nocollision = agent_j.pos + t_nocollision * agent_j.vel;
      var xj_collision   = agent_j.pos + t_collision * agent_j.vel;

      // Enforce collision free for x_collision using distance constraint
      var n = xi_collision - xj_collision;
      var d = length(n);

      var f = d - r;
      if (f < 0.0) {
        n = normalize(n);
        
        var k = k_longrange * exp(-t_nocollision*t_nocollision/t0);
        k = 1.0 - pow(1.0 - k, 1.0/(f32(itr + 1)));
        var w = 0.5;
        var dx = -w * f * n;

        // 4.5 Avoidance Model
        if (params.avoidance == 1.0f) {
          // get collision-free position
          xi_collision = xi_collision + dx;
          xj_collision = xj_collision - dx;

          // total relative displacement
          var d_vec = (xi_collision - xi_nocollision) - (xj_collision - xj_nocollision);

          // tangential relative displacement
          var d_tangent = d_vec - dot(d_vec, n)*n; 

          dx = dx + w * d_tangent;
          neighborCount = neighborCount + 1;
        }

        totalDx = totalDx + k * dx;
        neighborCount = neighborCount + 1;
      }
    }
  }

  if (neighborCount > 0) {
    agent.ppos = agent.ppos + avgCoeff * totalDx / f32(neighborCount);
  }

  // Write back
  agents_w.agents[index] = agent;
}