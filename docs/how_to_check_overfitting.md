# 如何判断模型是欠拟合还是过拟合

## 📊 方法一：使用分析脚本（推荐）

### 1. 安装依赖
```bash
pip install tensorboard matplotlib
```

### 2. 运行分析脚本
```bash
# 找到你的训练日志目录（通常在 runs/ 下）
python experiments/analyze_training_curves.py --logdir <你的日志目录>

# 例如:
python experiments/analyze_training_curves.py --logdir runs/lane/spin/20240101T120000_12345

# 保存图表
python experiments/analyze_training_curves.py --logdir runs/lane/spin/20240101T120000_12345 --save-plot training_curves.png

# 显示图表
python experiments/analyze_training_curves.py --logdir runs/lane/spin/20240101T120000_12345 --show-plot
```

### 3. 查看分析报告
脚本会自动分析并输出：
- ✅ 模型状态（欠拟合/过拟合/良好拟合）
- 📈 关键指标（训练集/验证集MAE、Gap）
- 🔄 收敛状态
- 💡 调整建议

## 📈 方法二：使用TensorBoard可视化

### 1. 启动TensorBoard
```bash
# 在项目根目录运行
tensorboard --logdir runs/

# 或者指定具体实验目录
tensorboard --logdir runs/lane/spin/
```

### 2. 在浏览器中打开
访问 `http://localhost:6006`

### 3. 查看关键指标
在TensorBoard中关注以下指标：
- `train_mae/epoch` vs `val_mae/epoch`
- `train_loss/epoch` vs `val_loss/epoch`

## 🔍 判断标准

### 过拟合 (Overfitting) 🔴
**特征：**
- 训练集MAE持续下降，但验证集MAE停止下降或开始上升
- 验证集MAE明显高于训练集MAE（Gap > 30%）
- 训练集和验证集曲线差距逐渐增大

**示例：**
```
Epoch 50: Train MAE = 0.15, Val MAE = 0.35  (Gap = 133%)
Epoch 60: Train MAE = 0.12, Val MAE = 0.38  (Gap = 217%)
Epoch 70: Train MAE = 0.10, Val MAE = 0.40  (Gap = 300%)
```

**解决方案：**
- 增加正则化 (`l2_reg: 1e-4`)
- 减少模型容量 (`hidden_size: 16 → 8`)
- 增加dropout（如果支持）
- 使用早停机制
- 增加数据增强

### 欠拟合 (Underfitting) 🟡
**特征：**
- 训练集和验证集MAE都很高且接近
- 训练集MAE下降很慢或几乎不下降
- 训练集和验证集曲线几乎重叠但都在高位

**示例：**
```
Epoch 50: Train MAE = 0.45, Val MAE = 0.47  (Gap = 4%)
Epoch 60: Train MAE = 0.44, Val MAE = 0.46  (Gap = 5%)
Epoch 70: Train MAE = 0.43, Val MAE = 0.45  (Gap = 5%)
```

**解决方案：**
- 增加模型容量 (`hidden_size: 16 → 32 → 64`)
- 增加层数 (`n_layers: 3 → 4 → 5`)
- 增加训练轮数 (`epochs: 80 → 150`)
- 增加学习率 (`lr: 0.0008 → 0.001`)
- 检查数据预处理

### 良好拟合 (Good Fit) 🟢
**特征：**
- 训练集和验证集MAE都在下降
- 验证集MAE略高于训练集（Gap < 10%）
- 两条曲线趋势一致，差距稳定

**示例：**
```
Epoch 50: Train MAE = 0.20, Val MAE = 0.22  (Gap = 10%)
Epoch 60: Train MAE = 0.18, Val MAE = 0.20  (Gap = 11%)
Epoch 70: Train MAE = 0.17, Val MAE = 0.19  (Gap = 12%)
```

## 📋 快速检查清单

### 检查训练日志
训练完成后，查看终端输出或日志文件，关注：
1. **最终指标对比**
   ```
   Train MAE: 0.XX
   Val MAE: 0.XX
   Test MAE: 0.XX
   ```

2. **训练过程**
   - 训练集loss是否持续下降？
   - 验证集loss是否也在下降？
   - 两者差距是否在增大？

### 检查TensorBoard日志位置
日志通常保存在：
```
runs/
  └── <dataset_name>/
      └── <model_name>/
          └── <experiment_name>/
              ├── events.out.tfevents.*
              └── config.yaml
```

### 常见问题

**Q: 找不到日志文件？**
A: 检查 `config.log_dir` 的设置，或者查看训练时的输出信息

**Q: TensorBoard显示空白？**
A: 确保指定了正确的日志目录，并且训练已经完成至少一个epoch

**Q: 如何比较不同实验？**
A: 在TensorBoard中，可以同时加载多个实验目录进行比较

## 🎯 实际案例

### 案例1：明显过拟合
```
训练曲线显示：
- Epoch 1-20: Train和Val MAE都在下降 ✅
- Epoch 21-40: Train继续下降，Val开始上升 ⚠️
- Epoch 41-60: Train持续下降，Val持续上升 🔴

判断：过拟合
建议：增加l2_reg到1e-4，或减少hidden_size
```

### 案例2：明显欠拟合
```
训练曲线显示：
- Epoch 1-80: Train和Val MAE都下降很慢
- 最终Train MAE = 0.50, Val MAE = 0.52 (都很高)
- 两条曲线几乎重叠

判断：欠拟合
建议：增加hidden_size到32或64，增加epochs到150
```

### 案例3：训练不充分
```
训练曲线显示：
- Epoch 1-5: Train和Val MAE都在快速下降
- 训练在第5个epoch就停止了（patience=5）

判断：训练被过早停止
建议：增加patience到20-30，让模型充分训练
```

## 💡 提示

1. **不要只看最终指标**：观察整个训练过程更重要
2. **关注趋势**：单个epoch的波动是正常的，要看整体趋势
3. **对比baseline**：如果模型效果不如均值baseline，很可能是欠拟合
4. **多次实验**：如果不确定，可以尝试调整参数后重新训练对比

