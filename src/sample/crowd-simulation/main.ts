import { makeSample, SampleInit } from '../../components/SampleLayout';
import headerWGSL from "./computeHeader.wgsl";
import spriteWGSL from './sprite.wgsl';
import blendVelocityWGSL from "./blendVelocity.wgsl";
import SRCollisionWGSL from "./SRCollision.wgsl";
import LRCollisionWGSL from "./LRCollision.wgsl";
import finalizeVelocityWGSL from "./finalizeVelocity.wgsl";

const init: SampleInit = async ({ canvas, pageState, stats }) => {
  // WebGPU device initialization
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported on this browser.');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }
  const hasTimestampQuery = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestampQuery ? ['timestamp-query'] : [],
  });

  // Canvas configuration
  const context = canvas.getContext("webgpu");
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
    deltaT: 0.01,
    stabilityIterations: 1,
    constraintIterations: 6,
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
            }
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

  var computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0, // params
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform"
        }
      },
      {
        binding: 1, // agents_read
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage"
        }
      },
      {
        binding: 2, // agents_write
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      }
    ]
  });

  const computePipelines = [];
  var pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [computeBindGroupLayout]
  });


  for(let i = 0; i < computeShaders.length; i++) {
    for (let itr = 0; itr < iterations[i]; itr++) {
      if (i == 2) // long range collision shader
        computePipelines.push( 
          device.createComputePipeline({
          layout: pipelineLayout,
          compute: {
            module: device.createShaderModule({
              code: computeShaders[i],
            }),
            entryPoint: 'main',
            constants: {
              1000 : itr + 1,
            }
          }})
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
          }})
        );
    }
  }

  const renderPassDescriptor = {
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
    -0.01, -0.02, 0.01,
    -0.02, 0.0, 0.02,
  ]);

  const spriteVertexBuffer = device.createBuffer({
    size: vertexBufferData.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(spriteVertexBuffer.getMappedRange()).set(vertexBufferData);
  spriteVertexBuffer.unmap();

  // pass parameters to the shaders (some params are omitted if not needed)
  const simParamBufferSize = 1 * Float32Array.BYTES_PER_ELEMENT;
  const simParamBuffer = device.createBuffer({
    size: simParamBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    simParamBuffer,
    0,
    new Float32Array([
      simParams.deltaT,
    ])
  );

  // can be updated with GUI, not implemented here (https://webgpu.github.io/webgpu-samples/samples/computeBoids#main.ts)

  const numAgents = 1500;
  const initialAgentData = new Float32Array(numAgents * 8);
  const goals = [
    [1.0, 1.0], [-1.0, -1.0]
  ];
  for (let i = 0; i < numAgents; ++i) {
    // position
    initialAgentData[8 * i + 0] = 2 * (Math.random() - 0.5);
    initialAgentData[8 * i + 1] = 2 * (Math.random() - 0.5);
    // velocity
    initialAgentData[8 * i + 2] = 2 * (Math.random() - 0.5) * 0.1;
    initialAgentData[8 * i + 3] = 2 * (Math.random() - 0.5) * 0.1;
    // planed / predicted position (initial value doesn't matter)
    initialAgentData[8 * i + 4] = initialAgentData[8 * i + 0]
    initialAgentData[8 * i + 5] = initialAgentData[8 * i + 1]
    // goal
    initialAgentData[8 * i + 6] = i % 2 == 0? goals[0][0] : goals[1][0];
    initialAgentData[8 * i + 7] = i % 2 == 0? goals[0][1] : goals[1][1];
  }

  const agentBuffers = new Array(2);
  const computeBindGroups = new Array(2);
  for (let i = 0; i < 2; ++i) {
    agentBuffers[i] = device.createBuffer({
      size: initialAgentData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Float32Array(agentBuffers[i].getMappedRange()).set(
      initialAgentData
    );
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

  let t = 0;
  var computeBindGroup = computeBindGroups[0];
  function switchBindGroup() {
    if (computeBindGroup == computeBindGroups[0])
      computeBindGroup = computeBindGroups[1];
    else if (computeBindGroup == computeBindGroups[1])
      computeBindGroup = computeBindGroups[0];
  }
  function getRenderBuffer() {
    if (computeBindGroup == computeBindGroups[0])
      return agentBuffers[0];
    else if (computeBindGroup == computeBindGroups[1])
      return agentBuffers[1];
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
        passEncoder.dispatchWorkgroups(Math.ceil(numAgents / 64));
        switchBindGroup();
      }

      passEncoder.end();
    }
    {
      const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
      passEncoder.setPipeline(renderPipeline);
      passEncoder.setVertexBuffer(0, getRenderBuffer());
      passEncoder.setVertexBuffer(1, spriteVertexBuffer);
      passEncoder.draw(3, numAgents, 0, 0);
      passEncoder.end();
    }

    device.queue.submit([commandEncoder.finish()]);

    ++t;
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
      'This example shows how to render and sample from a cubemap texture.',
    init,
    gui: true,
    stats: true,
    sources: [],
    filename: __filename,
  });

export default CrowdSimulation;
