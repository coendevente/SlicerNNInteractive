# 3D 语义选区布尔操作设计计划

## 背景

当前插件已经可以通过 nnInteractive 生成和迭代人体结构分割选区，并支持 point、bounding box、lasso、scribble 等交互提示。用户现在最需要的是在插件的 3D 视图中，对已经生成的语义化结构选区做进一步局部编辑，例如增加一部分、减去一部分、只保留交集，或者用另一个选区替换当前选区。

这类需求本质上是对当前 segment 的二值 mask 做集合运算。它不一定需要先扩展 nnInteractive server，因为 Slicer 客户端已经能读取、修改、写回当前 segment，并且已有同步机制可以把本地修改后的 segment 上传回 server，作为后续智能交互的初始状态。

## 设计目标

1. 允许用户在 Slicer 插件中直接编辑智能生成的结构选区。
2. 支持对语义化选区执行布尔操作：增加、减少、相交、替换。
3. 尽量复用 Slicer 原生 Segment Editor、Markups ROI、segment labelmap 等能力。
4. 本地布尔编辑后保持 nnInteractive server 状态同步，使后续 prompt 基于修正后的分割继续工作。
5. 先实现低风险、可测试的 MVP，再扩展更复杂的 3D 自由形状选区。

## 核心模型

设当前目标结构选区为 `S`，用户定义的操作选区为 `M`。

支持的布尔操作如下：

```text
增加:     S' = S OR M
减少/切割: S' = S AND NOT M
相交:     S' = S AND M
替换:     S' = M
```

计算得到 `S'` 后，通过现有 `show_segmentation()` 写回当前选中的 segment。随后更新本地状态，并把新 segment 上传到 server，保证下一次 nnInteractive 交互使用最新结果。

## 推荐架构

### 客户端本地执行

第一阶段应把布尔操作放在 Slicer 插件客户端执行，而不是新增 server API。

理由：

- 布尔操作只依赖二值 mask，不需要模型推理。
- 当前客户端已有 `get_segment_data()`、`show_segmentation()`、`upload_segment_to_server()`。
- 避免增加网络传输和 server 状态复杂度。
- 用户编辑后的 segment 可以继续作为 server 的 initial segmentation。

### Server 只负责智能交互

server 继续负责 nnInteractive 的 point、bbox、lasso、scribble 推理。客户端完成本地布尔编辑后，将结果作为当前 segment 上传给 server，让后续智能提示基于最新 mask 继续迭代。

## MVP 范围

第一版建议只做两个最实用的来源：

1. **segment-to-segment 布尔操作**
   - Target segment：当前要被修改的人体结构。
   - Operand segment：另一个语义化结构选区。
   - 例子：
     - 从肝脏 segment 中减去肿瘤 segment。
     - 把一个血管分支 segment 加入当前血管 segment。
     - 当前结构只保留和某个器官 segment 重叠的部分。

2. **ROI box 布尔切割**
   - 使用 `vtkMRMLMarkupsROINode` 定义 3D box。
   - 将 ROI 栅格化为操作 mask。
   - 典型操作是 `Subtract`，用于快速从当前结构中切掉一块。

这两个能力能覆盖最常见的“智能生成后人工增减一部分”的工作流，并且实现和测试成本可控。

## UI 设计建议

在现有插件 UI 中增加一个 `Selection Operations` 区域。

建议控件：

- Operation mode：
  - Add
  - Subtract
  - Intersect
  - Replace
- Operand source：
  - Segment
  - ROI
- Operand segment selector：
  - 当 source 为 Segment 时显示。
- ROI controls：
  - Create ROI
  - Apply ROI operation
  - Clear ROI
- Apply button：
  - 执行当前布尔操作。

默认操作建议为 `Subtract`，因为“从已生成结构中切掉错误部分”是最直接的修正需求。

## 代码实现计划

### 阶段 1：抽出 mask 布尔核心

新增一个本地方法，例如：

```python
def apply_boolean_operation(self, operation_mask, operation):
    current_mask = self.get_segment_data().astype(bool)
    operand_mask = operation_mask.astype(bool)

    if operation == "add":
        result_mask = current_mask | operand_mask
    elif operation == "subtract":
        result_mask = current_mask & ~operand_mask
    elif operation == "intersect":
        result_mask = current_mask & operand_mask
    elif operation == "replace":
        result_mask = operand_mask
    else:
        raise ValueError(f"Unknown operation: {operation}")

    self.show_segmentation(result_mask.astype(np.uint8))
    self.upload_segment_to_server()
```

注意点：

- `operation_mask` 必须和当前 volume shape 一致。
- 操作前应检查 source volume 和 segmentation geometry 是否匹配。
- 操作后应走现有 undo 机制，保证用户能撤销。

### 阶段 2：支持 segment-to-segment

实现流程：

1. 从 UI 获取 operand segment。
2. 使用 `slicer.util.arrayFromSegmentBinaryLabelmap()` 转为 numpy mask。
3. 调用 `apply_boolean_operation(mask, operation)`。
4. 保持 target segment 为当前选中 segment。

需要避免的情况：

- target segment 和 operand segment 是同一个 segment。
- operand segment 为空。
- operand segment 与当前 volume geometry 不一致。

### 阶段 3：支持 ROI box

实现流程：

1. 创建或复用一个 `vtkMRMLMarkupsROINode`。
2. 用户在 3D 视图中调整 ROI。
3. 根据 ROI 的 RAS bounds 和当前 volume 的 IJK transform 生成三维 mask。
4. 调用 `apply_boolean_operation(mask, operation)`。

第一版可以只支持轴对齐 ROI。旋转 ROI 或任意 closed surface 可放到后续阶段。

### 阶段 4：临时操作选区

新增隐藏或临时的 operation segmentation node，类似当前已有的 `ScribbleSegmentNode (do not touch)`。

用途：

- 用户使用 Segment Editor Paint、Scissors、Draw 等效果画出临时操作 mask。
- 临时 mask 不作为最终结构保存。
- 点击 Apply 后将临时 mask 和当前目标 segment 做布尔操作。
- Apply 后清空临时 mask。

这一步可以把 Slicer 原生 Segment Editor 的强大编辑能力变成“语义选区布尔操作”的输入来源。

## 与现有 nnInteractive 流程的关系

布尔编辑应被视为一次本地 segment 修改。

推荐行为：

1. 用户通过 nnInteractive 生成初始结构。
2. 用户执行本地布尔操作修正结构。
3. 插件写回当前 segment。
4. 插件上传当前 segment 到 server。
5. 用户继续用 point、bbox、lasso、scribble 做智能细化。

这样可以避免“本地看起来改了，但 server 仍然基于旧 mask 推理”的状态不一致问题。

## 测试计划

### 核心 mask 测试

构造简单 numpy mask，验证：

- Add 后 voxel 数量和位置正确。
- Subtract 后目标区域被移除。
- Intersect 后只保留重叠区域。
- Replace 后完全等于 operand mask。

### Slicer 回归测试

在 `slicer_plugin/SlicerNNInteractive/Testing/Python/` 中增加测试：

1. 创建两个简单 segment。
2. 选择一个作为 target，另一个作为 operand。
3. 执行布尔操作。
4. 使用 `arrayFromSegmentBinaryLabelmap()` 验证输出 mask。

ROI 测试可以先构造固定 ROI bounds，再检查输出 mask。

## 风险与注意事项

1. **几何一致性**
   - 不同 segment 或 volume 的 geometry 不一致时，mask 直接布尔运算会产生错误。
   - 第一版应强制使用当前 source volume 作为 reference。

2. **Undo 行为**
   - 布尔操作应调用现有 Segment Editor undo state。
   - `show_segmentation()` 当前已经调用 `saveStateForUndo()`，可以优先复用。

3. **3D surface 更新性能**
   - 大体积 mask 操作后重建 closed surface 可能较慢。
   - 现有 `show_segmentation()` 已检查 3D representation 是否显示，并只在需要时重建。

4. **Server 同步时机**
   - 每次 Apply 后立即上传最简单，但大体积可能慢。
   - 后续可以增加 “Apply locally” 和 “Sync to server” 的区分；MVP 先保持立即同步，减少状态歧义。

5. **UI 复杂度**
   - 不应一开始暴露过多高级选区工具。
   - 先把 segment-to-segment 和 ROI subtract 做稳定，再加自由形状操作选区。

## 推荐实施顺序

1. 新增布尔 mask 核心方法。
2. 增加 segment-to-segment UI 和逻辑。
3. 增加对应 Slicer 回归测试。
4. 增加 ROI box 操作。
5. 增加 ROI 回归测试。
6. 增加临时 operation mask 工作流。
7. 再考虑旋转 ROI、closed surface、free-form 3D selection 等高级能力。

## 结论

最合理的第一步不是扩展 nnInteractive server，而是在 Slicer 插件客户端增加本地布尔编辑层。这样可以快速满足“对智能生成的人体结构选区增加或减少一部分”的核心需求，同时保持后续智能交互能力。MVP 应从 segment-to-segment 布尔操作和 ROI box subtract 开始，之后再扩展到更复杂的 3D 自由形状选区。
