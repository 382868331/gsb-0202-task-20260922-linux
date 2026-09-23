# tps — 二维薄板样条（TPS）表面形变模块

面向离线医学配准工程的纯 Python 二维薄板样条（Thin-Plate Spline）fit/warp 库。
从零实现核心算法，仅使用 Python 标准库。

- **Python 版本**：3.14.7（语法上兼容 3.10+，但仅在 3.14.7 上验证）
- **依赖**：无第三方依赖，安装 Python 后即可运行；核心、测试、演示全部离线
- **平台**：Linux 原生

## 文件

| 文件 | 说明 |
|---|---|
| `tps.py` | 核心库：`fit_tps` / `TPS` / `kernel` 及错误类型 |
| `tests/test_tps.py` | unittest 测试（29 个用例） |
| `demo.py` | 固定输入演示：拟合形变、warp 网格、触发一次拒绝 |

## 运行

```bash
python -m unittest discover -s tests -v
python demo.py
```

## 数学模型

每个输出坐标独立求解：

```
f(p) = a0 + a1*x + a2*y + Σ_i w_i * U(|p - c_i|)
U(r) = r²·log(r)，U(0) = 0
```

系数 `[w, a]` 由分块线性系统解出（稠密高斯消元 + 部分主元，n+3 ≤ 43 阶）：

```
[ K + λI   P ] [ w ]   [ t ]
[   Pᵀ     0 ] [ a ] = [ 0 ]      P = [1, x, y]，K_ij = U(|c_i - c_j|)
```

底部块即侧条件 `Pᵀw = 0`。弯曲能为 `wᵀKw`（K 不含 λI），两个输出坐标分别计算并求和。

## 接口

```python
from tps import fit_tps, kernel, TPS, TPSError, TPSInputError, TPSDegenerateError, TPSSingularError

tps = fit_tps(source, target, lambda_=0.0)
```

- `fit_tps(source, target, lambda_=0.0) -> TPS`
  - `source` / `target`：`(x, y)` 对的可迭代对象，一一对应。
  - `lambda_`：实数 `>= 0`；`0` 为精确插值，越大越平滑。
- `TPS` 属性：
  - `control_points`、`targets`：合并重复后的控制点及对应目标
  - `weights`：`(w_x, w_y)`；`affine`：`((a0,a1,a2), (b0,b1,b2))`
  - `residuals`：每个控制点的 `(dx, dy)` 残差；`rms_error`、`max_error`
  - `bending_energy`、`bending_energy_per_axis`
- `TPS.warp(points) -> list[(x, y)]`、`TPS.warp_point(point) -> (x, y)`
- `TPS.jacobian(point) -> ((dfx/dx, dfx/dy), (dfy/dx, dfy/dy))`：解析 Jacobian，
  核导数项为 `(log(r²) + 1)·(p - c)`，在 `r = 0` 处为 0。
- `kernel(r)`：`U(r) = r²·log(r)`，`U(0) = 0`，`r` 须为有限实数且 `>= 0`。

## 输入上限与错误语义

- 控制点对数 `3 <= n <= 40`（上限按**输入原始对数**判定，在重复合并之前）。
- 所有数值输入必须是 `int`/`float`（**拒绝 `bool`**）且有限（**拒绝 NaN/Infinity**）；
  非法输入抛出带参数名/索引定位信息的 `TPSInputError`，不做静默纠正。
- 重复源点（坐标**精确相等**）：目标相同则合并为一点；目标不同抛 `TPSInputError`。
- 合并后不同控制点少于 3 个抛 `TPSInputError`；共线（仿射秩 < 3）抛 `TPSDegenerateError`。
- 通过退化检查但线性系统数值奇异（极端病态输入）抛 `TPSSingularError`。
- 拟合失败时不产生任何部分结果（`fit_tps` 要么返回完整 `TPS`，要么抛异常）。

## 误差口径

在坐标尺度为 O(1)~O(10) 的适定输入上（测试即按此口径断言）：

- `lambda_=0` 时控制点插值误差（`rms_error`/`max_error`）< 1e-8；
- 纯仿射目标下 `|w| < 1e-9`，弯曲能 < 1e-15，仿射系数与任意点 warp 误差 < 1e-8；
- 解析 Jacobian 与中心有限差分（h=1e-6）一致到 1e-5；
- 侧条件 `Pᵀw = 0` 残差 < 1e-8。

坐标尺度极端（如 1e±6 量级）时条件数恶化，精度相应下降，建议先归一化坐标。
测试容差针对上述良态尺度设定，不以宽容差掩盖边界错误。

## 设计取舍

- **精确相等判定重复点**：确定性、可解释；近似重复（浮点噪声）不合并，由 λ 正则化处理。
- **稠密高斯消元 O(n³)**：n ≤ 40 时规模 ≤ 43 阶，纯 Python 瞬时完成，无需任何数值库。
- **退化检查独立于求解器**：先以相对容差 1e-12 做仿射秩检验（共线/重合 → `TPSDegenerateError`），
  再求解；求解器主元失败单独报 `TPSSingularError`，两类失败不混淆。
- 不提供图像浏览器/可视化；demo 仅打印数值结果。

## 已知限制

- 仅二维、各向同性核 `U(r)=r²log(r)`；不实现 3D TPS、各向异性核或其他 RBF。
- 重复点判定为精确浮点相等，无容差合并。
- 未做坐标归一化；极端尺度输入需调用方自行预处理。
- 弯曲能按定义 `wᵀKw` 计算，未做量纲归一。
- 只实现本题契约的数学子集，不宣称与任何行业标准/第三方库兼容。
