import { makeSample, SampleInit } from '../../components/SampleLayout';
import headerWGSL from './wgsl/computeHeader.wgsl';
import spriteWGSL from './wgsl/sprite.wgsl';
import blendVelocityWGSL from './wgsl/blendVelocity.wgsl';
import SRCollisionWGSL from './wgsl/SRCollision.wgsl';
import LRCollisionWGSL from './wgsl/LRCollision.wgsl';
import finalizeVelocityWGSL from './wgsl/finalizeVelocity.wgsl';

const init: SampleInit = async ({ canvas, stats, gui }) => {
  // WebGPU device initialization
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported on this browser.');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No appropriate GPUAdapter found.');
  }
  const device = await adapter.requestDevice({});

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

  const simParams = {
    deltaT: 0.02,
    stabilityIterations: 1,
    constraintIterations: 6,
    numAgents: 4096, // MUST be power of 2
    avoidance: true,
    agentScale: 0,
  };
  simParams.agentScale =
    0.5 **
    Math.max(
      0.0,
      Math.floor(Math.log(simParams.numAgents / 512) / Math.log(4))
    );
  gui.add(simParams, 'numAgents');

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

  // create compute pipelines
  const computeShaders = [
    headerWGSL + blendVelocityWGSL,
    headerWGSL + SRCollisionWGSL,
    headerWGSL + LRCollisionWGSL,
    headerWGSL + finalizeVelocityWGSL,
  ];
  const iterations = [
    1,
    simParams.stabilityIterations,
    simParams.constraintIterations,
    1,
  ];

  const computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0, // params
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'uniform',
        },
      },
      {
        binding: 1, // agents_read
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        },
      },
      {
        binding: 2, // agents_write
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
        },
      },
    ],
  });

  const computePipelines = [];
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [computeBindGroupLayout],
  });

  for (let i = 0; i < computeShaders.length; i++) {
    for (let itr = 0; itr < iterations[i]; itr++) {
      if (i == 2)
        // long range collision shader
        computePipelines.push(
          device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
              module: device.createShaderModule({
                code: computeShaders[i],
              }),
              entryPoint: 'main',
              constants: {
                1000: itr + 1,
              },
            },
          })
        );
      else
        computePipelines.push(
          device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
              module: device.createShaderModule({
                code: computeShaders[i],
              }),
              entryPoint: 'main',
            },
          })
        );
    }
  }

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

  const computePassDescriptor = {};
  const vertexBufferData = new Float32Array([
    -0.01, -0.02, 0.01, -0.02, 0.0, 0.02,
  ]);
  for (let i = 0; i < vertexBufferData.length; i++) {
    vertexBufferData[i] *= simParams.agentScale;
  }

  const spriteVertexBuffer = device.createBuffer({
    size: vertexBufferData.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(spriteVertexBuffer.getMappedRange()).set(vertexBufferData);
  spriteVertexBuffer.unmap();

  // pass parameters to the shaders (some params are omitted if not needed)
  const simParamBufferSize = 3 * Float32Array.BYTES_PER_ELEMENT;
  const simParamBuffer = device.createBuffer({
    size: simParamBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    simParamBuffer,
    0,
    new Float32Array([simParams.deltaT, simParams.agentScale, simParams.avoidance? 1.0 : 0.0])
  );

  // can be updated with GUI, not implemented here (https://webgpu.github.io/webgpu-samples/samples/computeBoids#main.ts)
  const scene = {
    RANDOM: 0,
    SQUARE: 1,
    CIRCLE: 2,
  };
  const selectedScene = scene.SQUARE;

  const initialAgentData = new Float32Array(simParams.numAgents * 8);
  for (let i = 0; i < simParams.numAgents; ++i) {
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
      const power = Math.log2(simParams.numAgents / 2);
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

  const agentBuffers = new Array(2);
  const computeBindGroups = new Array(2);
  for (let i = 0; i < 2; ++i) {
    agentBuffers[i] = device.createBuffer({
      size: initialAgentData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Float32Array(agentBuffers[i].getMappedRange()).set(initialAgentData);
    agentBuffers[i].unmap();
  }

  for (let i = 0; i < 2; ++i) {
    computeBindGroups[i] = device.createBindGroup({
      layout: computeBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: simParamBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: agentBuffers[i],
            offset: 0,
            size: initialAgentData.byteLength,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: agentBuffers[(i + 1) % 2],
            offset: 0,
            size: initialAgentData.byteLength,
          },
        },
      ],
    });
  }

  let computeBindGroup = computeBindGroups[0];
  function switchBindGroup() {
    if (computeBindGroup == computeBindGroups[0])
      computeBindGroup = computeBindGroups[1];
    else if (computeBindGroup == computeBindGroups[1])
      computeBindGroup = computeBindGroups[0];
  }
  function getRenderBuffer() {
    if (computeBindGroup == computeBindGroups[0]) return agentBuffers[0];
    else if (computeBindGroup == computeBindGroups[1]) return agentBuffers[1];
  }

  function frame() {
    stats.begin();
    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    const commandEncoder = device.createCommandEncoder();
    {
      const passEncoder = commandEncoder.beginComputePass(
        computePassDescriptor
      );

      for (let i = 0; i < computePipelines.length; i++) {
        passEncoder.setPipeline(computePipelines[i]);
        passEncoder.setBindGroup(0, computeBindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(simParams.numAgents / 64));
        switchBindGroup();
      }

      passEncoder.end();
    }
    {
      const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
      passEncoder.setPipeline(renderPipeline);
      passEncoder.setVertexBuffer(0, getRenderBuffer());
      passEncoder.setVertexBuffer(1, spriteVertexBuffer);
      passEncoder.draw(3, simParams.numAgents, 0, 0);
      passEncoder.end();
    }

    device.queue.submit([commandEncoder.finish()]);
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
