export type SceneType = 'RANDOM' | 'SQUARE' | 'CIRCLE';

export function generateScene(
  scene: SceneType,
  numAgents: number,
  agentScale: number
): Float32Array {
  const initialAgentData = new Float32Array(numAgents * 8);
  for (let i = 0; i < numAgents; ++i) {
    if (scene == 'RANDOM') {
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
      initialAgentData[8 * i + 4] = initialAgentData[8 * i];
      initialAgentData[8 * i + 5] = initialAgentData[8 * i + 1];
      // goal
      initialAgentData[8 * i + 6] = i % 2 == 0 ? goals[0][0] : goals[1][0];
      initialAgentData[8 * i + 7] = i % 2 == 0 ? goals[0][1] : goals[1][1];
    } else if (scene == 'SQUARE') {
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
      initialAgentData[8 * i] =
        boundary[0][0] + x_idx * x_offset + perturbation[0];
      initialAgentData[8 * i + 1] =
        boundary[0][1] + y_idx * y_offset + perturbation[1];
      // velocity
      initialAgentData[8 * i + 2] = velocity[0] * agentScale;
      initialAgentData[8 * i + 3] = velocity[1] * agentScale;
      // planed / predicted position (initial value doesn't matter)
      initialAgentData[8 * i + 4] = initialAgentData[8 * i];
      initialAgentData[8 * i + 5] = initialAgentData[8 * i + 1];
      // goal
      initialAgentData[8 * i + 6] = goal[0];
      initialAgentData[8 * i + 7] = goal[1];
    }
  }
  return initialAgentData;
}
