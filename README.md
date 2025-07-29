## FEBID实验参数测量总结

### 表1：中心点沉积实验（修正版）

| 实验类型                  | 测量内容                     | 数学关系                                                                 |
|---------------------------|------------------------------|--------------------------------------------------------------------------|
| **中心点高度vs时间实验**<br>（固定电子束于一点，记录高度随时间变化） | **稳态沉积速率**<br>长时间线性段斜率 $A$<br>（$t > 3t_c$） | $$A = \Delta V \cdot \sigma \cdot f_0 \cdot n_{ss}$$<br>稳态覆盖度：<br>$$n_{ss} = \frac{k \Phi}{ \sigma f_0 + \frac{1}{\tau} + \frac{k \Phi}{n_0} }$$ |
|                           | **稳态偏移量**<br>长时间线性段截距 $B$ | $$B = \Delta V \cdot \sigma \cdot f_0 \cdot (n_{eq} - n_{ss}) \cdot t_c$$<br>平衡覆盖度：<br>$$n_{eq} = \frac{k \Phi}{ \frac{1}{\tau} + \frac{k \Phi}{n_0} }$$ |
|                           | **动力学特征时间**<br>过渡区曲线与稳态直线的偏差<br>（测量$\Delta h(t_1)$, $\Delta h(t_2)$等） | 特征时间计算：<br>$$t_c = \frac{t_2 - t_1}{\ln[\Delta h(t_1)/\Delta h(t_2)]}$$<br>理论表达式：<br>$$t_c = \frac{1}{ \sigma f_0 + \frac{1}{\tau} + \frac{k \Phi}{n_0} }$$ |
|                           | **初始沉积速率**<br>初始段斜率 $v_0$<br>（$t < 0.5t_c$） | $$v_0 = \Delta V \cdot \sigma \cdot f_0 \cdot n_{eq}$$ |

### 表2：形貌演化实验

| 实验类型 | 测量内容 | 数学关系 |
|----------|----------|----------|
| **不同时间点的形貌测量**<br>在$t=0,0.5,1,2,5,10 \times (r_b^2/D_{估计})$<br>等时间点测径向剖面 | 形貌开始偏离高斯的时间<br>($h_{center}$不再最高或<br>偏离高斯拟合>5%) | $t_{diffusion} = \frac{r_b^2}{D}$ |
| | 形貌不再变化的时间<br>(归一化形貌差异<2%) | $t_{steady} \approx 3 \cdot t_{diffusion}$ |
| | 稳态时$h_{center}/h_{max}$比值 | $\Gamma < 0.3$: $h_{center}/h_{max} \approx 1$<br>$0.3 < \Gamma < 3$: $0.8 < h_{center}/h_{max} < 0.95$<br>$\Gamma > 3$: $h_{center}/h_{max} < 0.8$<br>其中 $\Gamma = \frac{\sigma \cdot f_0 \cdot r_b^2}{D}$ |
| | 火山口半径$r_{crater}$<br>(最高点到中心距离) | $r_{crater} \approx r_b \cdot \sqrt{\ln(\Gamma)}$<br>(适用于$\Gamma > 3$) |


