import { vec2 } from 'gl-matrix';
import { makeSample, SampleInit } from '../../components/SampleLayout';
import spriteWGSL from './wgsl/sprite.wgsl';

const init: SampleInit = async ({ canvas, stats }) => {
  // WebGPU device initialization
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported on this browser.');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No appropriate GPUAdapter found.');
  }
  const hasTimestampQuery = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestampQuery ? ['timestamp-query'] : [],
  });

  // Canvas configuration
  const context = canvas.getContext('webgpu');
  const devicePixelRatio = window.devicePixelRatio;
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
    alphaMode: 'premultiplied',
  });

  const numAgents = 1024; // MUST be power of 2
  const simParams = {
    deltaT: 0.02,
    stabilityIterations: 1,
    constraintIterations: 6,
    agentScale:
      0.5 ** Math.max(0.0, Math.floor(Math.log(numAgents / 512) / Math.log(4))),
  };

  // shaders
  const spriteShaderModule = device.createShaderModule({ code: spriteWGSL });
  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: spriteShaderModule,
      entryPoint: 'vert_main',
      buffers: [
        {
          // instanced particles buffer
          arrayStride: 8 * 4,
          stepMode: 'instance',
          attributes: [
            {
              // instance position
              shaderLocation: 0,
              offset: 0,
              format: 'float32x2',
            },
            {
              // instance velocity
              shaderLocation: 1,
              offset: 2 * 4,
              format: 'float32x2',
            },
            {
              // instance planed position
              shaderLocation: 3,
              offset: 4 * 4,
              format: 'float32x2',
            },
            {
              // instance goal
              shaderLocation: 4,
              offset: 6 * 4,
              format: 'float32x2',
            },
          ],
        },
        {
          // vertex buffer
          arrayStride: 2 * 4,
          stepMode: 'vertex',
          attributes: [
            {
              // vertex positions
              shaderLocation: 2,
              offset: 0,
              format: 'float32x2',
            },
          ],
        },
      ],
    },
    fragment: {
      module: spriteShaderModule,
      entryPoint: 'frag_main',
      targets: [
        {
          format: canvasFormat,
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: undefined, // Assigned later
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  };
  const vertexBufferData = new Float32Array([
    -0.01, -0.02, 0.01, -0.02, 0.0, 0.02,
  ]);

  const spriteVertexBuffer = device.createBuffer({
    size: vertexBufferData.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(spriteVertexBuffer.getMappedRange()).set(vertexBufferData);
  spriteVertexBuffer.unmap();

  const scene = {
    RANDOM: 0,
    SQUARE: 1,
    CIRCLE: 2,
  };
  const selectedScene = scene.SQUARE;

  let initialAgentData = new Float32Array(numAgents * 8);
  for (let i = 0; i < numAgents; ++i) {
    if (selectedScene == scene.RANDOM) {
      const goals = [
        [1.0, 1.0],
        [-1.0, -1.0],
      ];
      // position
      initialAgentData[8 * i + 0] = 2 * (Math.random() - 0.5);
      initialAgentData[8 * i + 1] = 2 * (Math.random() - 0.5);
      // velocity
      initialAgentData[8 * i + 2] = 2 * (Math.random() - 0.5) * 0.1;
      initialAgentData[8 * i + 3] = 2 * (Math.random() - 0.5) * 0.1;
      // planed / predicted position (initial value doesn't matter)
      initialAgentData[8 * i + 4] = initialAgentData[8 * i + 0];
      initialAgentData[8 * i + 5] = initialAgentData[8 * i + 1];
      // goal
      initialAgentData[8 * i + 6] = i % 2 == 0 ? goals[0][0] : goals[1][0];
      initialAgentData[8 * i + 7] = i % 2 == 0 ? goals[0][1] : goals[1][1];
    } else if (selectedScene == scene.SQUARE) {
      const y_margin = 0.2;
      const x_margin = 0.2;
      // boundary is a bounding box represented by [xMin, yMin], [xMax, yMax]
      const topBoundary = [
        [x_margin - 1.0, y_margin],
        [1.0 - x_margin, 1.0 - y_margin],
      ];
      const botBoundary = [
        [x_margin - 1.0, y_margin - 1.0],
        [1.0 - x_margin, -y_margin],
      ];
      const boundary = i % 2 == 0 ? topBoundary : botBoundary;
      const velocity = i % 2 == 0 ? [0, -0.1] : [0, 0.1];
      const goal = i % 2 == 0 ? [0, -1.0] : [0, 1.0];
      const Idx = Math.floor(i / 2);
      const power = Math.log2(numAgents / 2);
      const x_count = 2 ** Math.ceil(power / 2);
      const y_count = 2 ** Math.floor(power / 2);
      const x_offset = (boundary[1][0] - boundary[0][0]) / x_count;
      const y_offset = (boundary[1][1] - boundary[0][1]) / y_count;
      const x_idx = Idx % x_count;
      const y_idx = Math.floor(Idx / x_count);
      // position
      const randomVal = 2 * (Math.random() - 0.5);
      const perturbation = [
        (x_offset * randomVal) / 5.0,
        (y_offset * randomVal) / 5.0,
      ];
      initialAgentData[8 * i + 0] =
        boundary[0][0] + x_idx * x_offset + perturbation[0];
      initialAgentData[8 * i + 1] =
        boundary[0][1] + y_idx * y_offset + perturbation[1];
      // velocity
      initialAgentData[8 * i + 2] = velocity[0] * simParams.agentScale;
      initialAgentData[8 * i + 3] = velocity[1] * simParams.agentScale;
      // planed / predicted position (initial value doesn't matter)
      initialAgentData[8 * i + 4] = initialAgentData[8 * i + 0];
      initialAgentData[8 * i + 5] = initialAgentData[8 * i + 1];
      // goal
      initialAgentData[8 * i + 6] = goal[0];
      initialAgentData[8 * i + 7] = goal[1];
    }
  }

  function buffer2ObjArray(buffer) {
    const objArray = [];
    for (let i = 0; i < numAgents; ++i) {
      const agent = {
        pos: vec2.fromValues(buffer[8 * i + 0], buffer[8 * i + 1]),
        vel: vec2.fromValues(buffer[8 * i + 2], buffer[8 * i + 3]),
        ppos: vec2.fromValues(buffer[8 * i + 4], buffer[8 * i + 5]),
        goal: vec2.fromValues(buffer[8 * i + 6], buffer[8 * i + 7]),
      };
      objArray.push(agent);
    }
    return objArray;
  }

  function objArray2Buffer(objArray) {
    const agentBuffer = new Float32Array(numAgents * 8);
    for (let i = 0; i < numAgents; ++i) {
      // position
      agentBuffer[8 * i + 0] = objArray[i].pos[0];
      agentBuffer[8 * i + 1] = objArray[i].pos[1];
      // velocity
      agentBuffer[8 * i + 2] = objArray[i].vel[0];
      agentBuffer[8 * i + 3] = objArray[i].vel[1];
      // planed / predicted position
      agentBuffer[8 * i + 4] = objArray[i].ppos[0];
      agentBuffer[8 * i + 5] = objArray[i].ppos[1];
      // goal
      agentBuffer[8 * i + 6] = objArray[i].goal[0];
      agentBuffer[8 * i + 7] = objArray[i].goal[1];
    }
    return agentBuffer;
  }

  const agentBuffer = device.createBuffer({
    size: initialAgentData.byteLength,
    usage:
      GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(agentBuffer.getMappedRange()).set(initialAgentData);
  agentBuffer.unmap();

  function updateAgents(agents_a, agents_b) {
    const scale = simParams.agentScale;
    const agentSpeed = 0.1 * scale;
    const nearRadius = 0.2 * scale; // threshold for SR collision
    const farRadius = 0.5 * scale; // threshold for LR collision
    const cohesionRadius = 1.0 * scale; // threshold for cohesion
    const agentRadius = 0.03 * scale; // agent size
    const blendFactor = 0.0385;
    const k_shortrange = 1.0;
    const k_longrange = 0.15;
    const avgCoeff = 1.2;
    const eps = 0.0001;
    const t0 = 20.0;

    let agents_r = agents_a;
    let agents_w = agents_b;
    function swap() {
      [agents_r, agents_w] = [agents_w, agents_r];
    }

    // stage1. blendVelocity
    for (let i = 0; i < numAgents; ++i) {
      const agent = { ...agents_r[i] };

      const directionToGoal = vec2.create();
      vec2.subtract(directionToGoal, agent.goal, agent.pos);
      vec2.normalize(directionToGoal, directionToGoal);

      // Calculate the new velocity vector
      const vp = vec2.create();
      vec2.scale(vp, directionToGoal, agentSpeed);

      // Update the velocity using a blend factor
      vec2.scaleAndAdd(
        agent.vel,
        vec2.scale(vec2.create(), vp, blendFactor),
        agent.vel,
        1.0 - blendFactor
      );

      // Update the previous position based on the new velocity
      const deltaVel = vec2.create();
      vec2.scale(deltaVel, agent.vel, simParams.deltaT);
      vec2.add(agent.ppos, agent.pos, deltaVel);

      agents_w[i] = agent;
    }
    swap();

    // stage2. SR Collision
    for (let itr = 0; itr < simParams.stabilityIterations; itr++) {
      for (let i = 0; i < numAgents; ++i) {
        const agent = { ...agents_r[i] };

        const totalDx = vec2.create();
        let neighborCount = 0;

        for (let j = 0; j < numAgents; j++) {
          if (i == j) {
            continue;
          }

          const agent_j = agents_r[j];
          const n = vec2.create();
          vec2.subtract(n, agent.ppos, agent_j.ppos);
          const d = vec2.length(n);

          if (d > nearRadius) {
            continue;
          }

          const f = d - 2.0 * agentRadius; // assume all agents have the same size

          if (f < 0.0) {
            // 4.2 Short Range Collision
            vec2.normalize(n, n);
            const w = 0.5; // assume all agents have the same weight
            const dx = vec2.create();
            vec2.scale(dx, n, -w * k_shortrange * f);

            // 4.2 Friction (not implemented yet)

            vec2.add(totalDx, totalDx, dx);
            neighborCount++;
          }
        }

        if (neighborCount > 0) {
          vec2.scale(totalDx, totalDx, avgCoeff / neighborCount);

          vec2.add(agent.pos, agent.pos, totalDx);
          vec2.add(agent.ppos, agent.ppos, totalDx);
        }
        agents_w[i] = agent;
      }
      swap();
    }

    // stage3. LR Collision
    for (let itr = 0; itr < simParams.constraintIterations; itr++) {
      for (let i = 0; i < numAgents; ++i) {
        const agent = { ...agents_r[i] };

        const totalDx = vec2.create();
        let neighborCount = 0;

        for (let j = 0; j < numAgents; j++) {
          if (i == j) {
            continue;
          }

          const agent_j = agents_r[j];
          const d = vec2.distance(agent.pos, agent_j.pos);
          if (d > farRadius) {
            continue;
          }

          const f = d - 2.0 * agentRadius; // assume all agents have the same size
          if (f < 0.0) {
            // 4.4 Long Range Collision
            const r = 2.0 * agentRadius;
            let r2 = r * r;
            const dt = simParams.deltaT;

            const dist = vec2.distance(agent.pos, agent_j.pos);
            if (dist < r) {
              r2 = (r - dist) * (r - dist);
            }

            const x_ij = vec2.subtract(vec2.create(), agent.pos, agent_j.pos); // relative displacement
            const v_ij = vec2.scale(
              // relative velocity
              vec2.create(),
              vec2.subtract(
                vec2.create(),
                vec2.subtract(vec2.create(), agent.ppos, agent.pos),
                vec2.subtract(vec2.create(), agent_j.ppos, agent_j.pos)
              ),
              1.0 / dt
            );

            const a = vec2.dot(v_ij, v_ij);
            const b = -vec2.dot(x_ij, v_ij);
            const c = vec2.dot(x_ij, x_ij) - r2;
            const discr = b * b - a * c;
            if (discr < 0.0 || Math.abs(a) < eps) {
              continue;
            }

            const sqrtDiscr = Math.sqrt(discr);

            // Compute exact time to collision
            const t = (b - sqrtDiscr) / a;

            // Prune out invalid case
            if (t < eps || t > t0) {
              continue;
            }

            // Get time before and after collision
            const t_nocollision = dt * Math.floor(t / dt);
            const t_collision = dt + t_nocollision;

            // Get collision and collision-free positions
            const xi_collision = vec2.add(
              vec2.create(),
              agent.pos,
              vec2.scale(vec2.create(), agent.vel, t_collision)
            );
            const xj_collision = vec2.add(
              vec2.create(),
              agent_j.pos,
              vec2.scale(vec2.create(), agent_j.vel, t_collision)
            );

            // Enforce collision-free for x_collision using distance constraint
            const n = vec2.subtract(vec2.create(), xi_collision, xj_collision);
            const dCollision = vec2.length(n);

            const fCollision = dCollision - r;
            if (fCollision < 0.0) {
              vec2.normalize(n, n);

              const k =
                k_longrange * Math.exp((-t_nocollision * t_nocollision) / t0);
              const kAdjusted = 1.0 - Math.pow(1.0 - k, 1.0 / (itr + 1));
              const w = 0.5;
              const dx = vec2.scale(vec2.create(), n, -w * fCollision);

              // 4.5 Avoidance Model (not implemented)

              vec2.scaleAndAdd(totalDx, totalDx, dx, kAdjusted);
              neighborCount++;
            }
          }
        }

        if (neighborCount > 0) {
          vec2.add(
            agent.ppos,
            agent.ppos,
            vec2.scale(vec2.create(), totalDx, avgCoeff / neighborCount)
          );
        }

        agents_w[i] = agent;
      }
      swap();
    }

    // stage4. Finalize Velocity and Cohesion
    for (let i = 0; i < numAgents; ++i) {
      const agent = { ...agents_r[i] };

      function poly6Kernel(r) {
        let w = 0.0;
        const xsph_c = 217.0;
        if (eps <= r && r <= xsph_c) {
          w = 315.0 / (64.0 * Math.PI * Math.pow(xsph_c, 9.0));
          const hmr = xsph_c * xsph_c - r * r;
          w = w * hmr * hmr * hmr;
        }
        return w;
      }

      vec2.scale(
        agent.vel,
        vec2.subtract(vec2.create(), agent.ppos, agent.pos),
        1.0 / simParams.deltaT
      );

      // 4.3 Cohesion (adding XSPH viscosity)
      const avgVel = vec2.create();

      for (let j = 0; j < numAgents; j++) {
        if (i === j) {
          continue;
        }

        const agent_j = agents_r[j];
        const equality = vec2.exactEquals(agent_j.goal, agent.goal);
        if (!equality[0] || !equality[1]) {
          continue;
        }

        const d = vec2.distance(agent.ppos, agent_j.ppos);
        if (d > cohesionRadius) {
          continue;
        }
        const w = poly6Kernel(d * d);
        vec2.scaleAndAdd(
          avgVel,
          avgVel,
          vec2.subtract(vec2.create(), agent.vel, agent_j.vel),
          w
        );
      }

      const xsph_h = 7.0;
      vec2.scaleAndAdd(agent.vel, agent.vel, avgVel, xsph_h);

      // 4.6 Maximum Speed and Acceleration Limiting
      const dir = vec2.normalize(vec2.create(), agent.vel);
      const maxSpeed = 1.2 * agentSpeed;

      if (vec2.length(agent.vel) > maxSpeed) {
        vec2.scale(agent.vel, dir, maxSpeed);
      }

      // Finally, update position
      vec2.scaleAndAdd(agent.pos, agent.pos, agent.vel, simParams.deltaT);

      agents_w[i] = agent;
    }

    return agents_w;
  }
  function frame() {
    stats.begin();
    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    const commandEncoder = device.createCommandEncoder();
    {
      // compute pass through raw javascript
      const agents_r = buffer2ObjArray(initialAgentData);
      const agents_w = buffer2ObjArray(initialAgentData);
      const updated_agents = updateAgents(agents_r, agents_w);
      initialAgentData = objArray2Buffer(updated_agents);

      // write the updated CPU agent buffer to the GPU
      device.queue.writeBuffer(agentBuffer, 0, initialAgentData);
    }
    {
      const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
      passEncoder.setPipeline(renderPipeline);
      passEncoder.setVertexBuffer(0, agentBuffer);
      passEncoder.setVertexBuffer(1, spriteVertexBuffer);
      passEncoder.draw(3, numAgents, 0, 0);
      passEncoder.end();
    }

    device.queue.submit([commandEncoder.finish()]);
    // console.log(t);
    stats.end();
    requestAnimationFrame(frame);
  }

  // This effectively the main loop but in a recursive fashion:
  // we schedule next frame at the end of each frame call. We only
  // need to call requestAnimationFrame(frame) once, then it will
  // recursively call itself.
  requestAnimationFrame(frame);
};

const CrowdSimulation: () => JSX.Element = () =>
  makeSample({
    name: 'Crowd Simulation',
    description:
      'This example shows an algorithm to simulate crowd movement and interaction (see )',
    init,
    gui: true,
    stats: true,
    sources: [],
    filename: __filename,
  });

export default CrowdSimulation;
