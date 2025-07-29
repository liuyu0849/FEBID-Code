### 表1：中心点沉积实验

| 实验类型 | 测量内容 | 数学关系 |
|----------|----------|----------|
| **中心点高度vs时间实验**<br>固定电子束于一点<br>记录高度随时间变化 | 长时间线性段斜率A<br>(t > 3t_c) | $A = \Delta V \cdot \sigma \cdot f_0 \cdot n_{ss}$<br>其中 $n_{ss} = \frac{k \cdot \Phi}{1/\tau + k \cdot \Phi/n_0 + \sigma \cdot f_0}$ |
| | 长时间线性段截距B | $B = \Delta V \cdot \sigma \cdot f_0 \cdot (n_{eq} - n_{ss}) \cdot t_c$<br>其中 $n_{eq} = \frac{k \cdot \Phi}{1/\tau + k \cdot \Phi/n_0}$ |
| | 过渡区曲线与稳态直线的偏差<br>$\Delta h(t_1)$, $\Delta h(t_2)$等多个点 | $t_c = \frac{t_2 - t_1}{\ln[\|\Delta h(t_1)\|/\|\Delta h(t_2)\|]}$<br>其中 $t_c = \frac{1}{k \cdot \Phi/n_0 + 1/\tau + \sigma \cdot f_0}$ |
| | **初始沉积速率**<br>初始段斜率 $v_0$<br> | $$v_0 = \Delta V \cdot \sigma \cdot f_0 \cdot n_{eq}$$ |


### 表2：形貌演化实验

以下是FEBID形貌参数的总结表格：

| **参数**                | **公式/条件**                          | **物理意义/说明**                                                                 |
|-------------------------|----------------------------------------|---------------------------------------------------------------------------------|
| **γ的定义**             | $\gamma = \frac{L_{\text{eff}}}{r_b} = \frac{ \sqrt{ \frac{D}{1/\tau + \sigma \cdot f_0} } }{r_b}$ | $L_{\text{eff}}$：有效扩散长度（前驱体扩散特征距离）<br>$r_b$：电子束斑半径<br>**物理意义**：扩散补充能力与束斑尺寸的比值 |
| **火山口形貌条件**      | $\gamma < 0.3$                         | 扩散长度远小于束斑尺寸（$L_{\text{eff}} < 0.3 \times r_b$）                     |
| **高斯→火山口转变时间，即hmax不再是中心点的时间** | $t_{\text{transition}} = \frac{r_b^2}{D}$ | **扩散特征时间**<br>与γ无关（火山口形貌下）<br>**物理意义**：前驱体扩散穿越束斑所需时间 |
| **火山口半径**          | $r_{\text{crater}} = r_b \cdot \sqrt{-2 \ln \gamma}$ | **条件**：仅当 $\gamma < 0.3$ 时存在<br>**例**：$\gamma=0.1$ 时 $r_{\text{crater}} \approx 2.1 r_b$<br>**趋势**：半径随γ减小而增大 |
