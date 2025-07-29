### 表1：中心点沉积实验

| 实验类型 | 测量内容 | 数学关系 |
|----------|----------|----------|
| **中心点高度vs时间实验**<br>固定电子束于一点<br>记录高度随时间变化 | 长时间线性段斜率A<br>(t > 3t_c) | $A = \Delta V \cdot \sigma \cdot f_0 \cdot n_{ss}$<br>其中 $n_{ss} = \frac{k \cdot \Phi}{1/\tau + k \cdot \Phi/n_0 + \sigma \cdot f_0}$ |
| | 长时间线性段截距B | $B = \Delta V \cdot \sigma \cdot f_0 \cdot (n_{eq} - n_{ss}) \cdot t_c$<br>其中 $n_{eq} = \frac{k \cdot \Phi}{1/\tau + k \cdot \Phi/n_0}$ |
| | 过渡区曲线与稳态直线的偏差<br>$\Delta h(t_1)$, $\Delta h(t_2)$等多个点 | $t_c = \frac{t_2 - t_1}{\ln[\|\Delta h(t_1)\|/\|\Delta h(t_2)\|]}$<br>其中 $t_c = \frac{1}{k \cdot \Phi/n_0 + 1/\tau + \sigma \cdot f_0}$ |
| | **初始沉积速率**<br>初始段斜率 $v_0$<br> | $$v_0 = \Delta V \cdot \sigma \cdot f_0 \cdot n_{eq}$$ |


### 表2：形貌演化实验

| 实验类型 | 测量内容 | 数学关系 |
|----------|----------|----------|
| **不同时间点的形貌测量**<br>在$t=0,0.5,1,2,5,10 \times (r_b^2/D_{估计})$<br>等时间点测径向剖面 | 形貌开始偏离高斯的时间<br>($h_{center}$不再最高或<br>偏离高斯拟合>5%) | $t_{diffusion} = \frac{r_b^2}{D}$ |
| | 形貌不再变化的时间<br>(归一化形貌差异<2%) | $t_{steady} \approx 3 \cdot t_{diffusion}$ |
| | 稳态时$h_{center}/h_{max}$比值 | $\Gamma < 0.3$: $h_{center}/h_{max} \approx 1$<br>$0.3 < \Gamma < 3$: $0.8 < h_{center}/h_{max} < 0.95$<br>$\Gamma > 3$: $h_{center}/h_{max} < 0.8$<br>其中 $\Gamma = \frac{\sigma \cdot f_0 \cdot r_b^2}{D}$ |
| | 火山口半径$r_{crater}$<br>(最高点到中心距离) | $r_{crater} \approx r_b \cdot \sqrt{\ln(\Gamma)}$<br>(适用于$\Gamma > 3$) |


