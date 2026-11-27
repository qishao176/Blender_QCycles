# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Translations for the optional photon caustics and GPU guiding controls.

Keep the English RNA identifiers and the user's language preferences unchanged.
A separate context prevents short method names from reusing unrelated mesh terms.
"""

import bpy

CONTEXT = "CyclesPhotonCaustics"

_GPU_GUIDING_ZH = {
    'GPU Surface Path Guiding': 'GPU 表面路径引导',
    'GPU Guiding Training Samples': 'GPU 引导训练采样',
    'Training Samples': '训练样本',
    'GPU Guiding Probability': 'GPU 引导概率',
    'Learn a bounded directional distribution on NVIDIA GPUs for reflective surfaces; photon emission keeps its own guiding': '在 NVIDIA GPU 上为反射表面学习容量受限的方向分布；光子发射继续使用自身引导',
    'Initial camera samples used to update the guide, then freeze it; included in the render sample budget': '使用前期相机采样更新引导，随后冻结；训练计入渲染采样预算',
    'Fraction sampled from the learned guide; the remaining fraction uses the original BSDF': '从学习分布中抽样的比例，其余使用原有 BSDF 抽样',
}

_HDR_PHOTON_ZH = {
    "Photon Settings Advice": "光子设置建议",
    "Probe status": "状态",
    "Waiting for IPR data": "等待 IPR 数据",
    "Collecting observations": "正在收集数据",
    "Sampling": "采样中",
    "Probed rounds": "已抽查轮数",
    "Emitted per device / round": "每台设备每轮发射数",
    "Surface deposits / round": "每台设备每轮表面落点",
    "Volume deposits / round": "每台设备每轮体积落点",
    "Deposit rate": "入图率",
    "Emission rounds / budget": "已发射轮数 / 预算",
    "Isolated / tested": "孤立落点 / 抽查数",
    "Subpixel / in view": "亚像素足迹 / 视野内抽查数",
    "Helped by widening / isolated": "展宽后有邻居 / 孤立落点",
    "Deposit count variation": "落点数变异系数",
    "Suggested gather radius": "建议聚集半径",
    "Suggested photons / round": "建议每轮光子数",
    "Suggested rounds / samples": "建议轮数 / 采样数",
    "Check light and caster": "检查灯光和投射物体",
    "Check beam coverage": "检查体积光束覆盖",
    "No volume hits yet": "尚无体积落点",
    "Sparse surface coverage": "表面落点稀疏",
    "Surface kernels overlap": "表面光子核已交叠",
    "Mixed surface coverage": "表面覆盖不均",
    "At capacity": "已达发射上限",
    'HDR Photon Source': 'HDR 光子发射源',
    'Photon Source Contrast': '光源检测阈值',
    'Photon Source Radius': '光源范围倍率',
    'Show Photon Source': '显示光子发射范围',
    'Minimum peak brightness relative to the World average for HDR photon emission; lower values accept weaker bright regions': 'HDR 峰值亮度相对环境平均亮度的最低倍数；降低可接受较弱的明亮区域',
    'Scale the detected photon source angular radius, up to 90 degrees; emission evaluates the original World shader with its sampling PDF': '缩放检测区域的角半径，最大 90 度；发射按原环境着色器和抽样概率计算能量',
    'Draw the actual detected World photon cone as a red viewport overlay during rendered preview': '在渲染预览中用红圈叠加显示实际光子发射区域，不进入最终渲染',
}

_LABELS_ZH = {
    "Use Relative Motion Blur": "修正为相对运动模糊",
    "Correct Photon Motion Blur": "修正光子运动模糊",
    "Deep Photon Depth Layers": "Deep 光子深度分层",
    "Depth resolution for volume photon lighting in Deep EXR; more layers preserve depth detail and use more memory": "Deep EXR 中体积光子光照的深度分辨率；更多层保留更多深度细节并增加内存占用",

    "Photon Strength": "光子强度",
    "Photon Size": "光子尺寸系数",
    "Scale only this light's photon contribution; 1 matches the physical light power": "仅缩放此灯光的光子贡献；1 对应灯光的物理功率",
    "Scale this light's final surface and volume photon kernels after reconstruction, preserving their integrated energy": "在重建后缩放此灯光的表面及体积光子核，保持核的积分能量",

    **_HDR_PHOTON_ZH,
    **_GPU_GUIDING_ZH,
    'Compatibility Neighbor Selection': '兼容性邻域选择',
    'Neighbor Evaluations': '邻域求值数',
    'OptiX Photon SER': 'OptiX 光子着色执行重排',
    'Sample ellipse reconstruction statistics from compatible neighbors with probability compensation. Adds variance; does not discard photon illumination or change LPE membership': '按兼容性抽样椭圆重建统计，并补偿选择概率。会增加统计方差，不丢弃光子光照，也不改变 LPE 归属',
    'Weighted evaluations from up to 32 pilot candidates per photon; sparse neighborhoods are evaluated exactly': '每个光子从最多 32 个统计候选中加权抽样；稀疏邻域完整求值',
    'Reorder photon material shading on supported OptiX devices. Performance depends on shader diversity and live state; CPU and CUDA are unaffected': '在支持的 OptiX 设备上重排光子材质着色。性能取决于材质差异与存活状态量；不影响 CPU 和 CUDA',

    'Reconstruction Method': '重建方法',
    'Local Scalar': '局部等比例',
    'Regional Coverage': '区域覆盖',
    'Regional Anisotropic': '区域各向异性',
    'Adaptive Ellipse Multiplier': '椭圆自适应乘数',
    'Statistical Radius Multiplier': '统计半径乘数',
    'Previous Interface Continuity': '上一界面连续性',
    'Reconstruction Neighbors': '重建邻域',
    'Same Light and Receiver': '同灯光与接收面',
    'Last Interface Normal': '上一界面法线',
    'Full Caster Path': '完整投射路径',
    'Caster Path and Normal': '投射路径与法线',
    'Interface Normal Angle': '界面法线夹角',
    "Force Zero Roughness for Photons": "强制光子粗糙度为零",
    "Use zero roughness for caustic photon reflection, transmission and coat. Enable on the object for all its materials or on the material for every object using it. Disable both to use shader roughness": "焦散光子的反射、折射及清漆反射按零粗糙度计算。物体开关作用于此物体上的全部材质，材质开关作用于所有使用此材质的物体。任一开启即可生效，两处均关闭时使用着色器粗糙度",
    "Ignore material Normal Map and Bump for photon casting, keeping smooth and custom mesh normals and true displacement. Enable on the object for all its materials or on the material for every object using it. Disable both to use shader normals": "投射光子时忽视材质法线贴图与凹凸，保留网格平滑法线、自定义法线和真实置换。物体开关作用于此物体上的全部材质，材质开关作用于所有使用此材质的物体。任一开启即可生效，两处均关闭时使用着色器法线",
    "Force Geometry Normals": "强制使用几何法线",
    "Ignore Material Normals for Photons": "光子忽视材质法线输出",
    "Reflective Caustics": "反射焦散",
    "Refractive Caustics": "折射焦散",
    "Allow this material to reflect shadow caustics on objects with Cast Shadow Caustics enabled": "允许此材质在开启投射阴影焦散的物体上产生反射焦散",
    "Allow this material to transmit shadow caustics on objects with Cast Shadow Caustics enabled": "允许此材质在开启投射阴影焦散的物体上产生折射焦散",
    "Adaptive Photon Reconstruction": "光子自适应重建",
    "Adaptive Photon Radius": "自适应光子半径",
    "Gradient Protection": "梯度保护",
    "Target Photon Neighbors": "目标邻域光子数",
    "Maximum Radius Multiplier": "最大半径乘数",
    "Gradient Protection Strength": "梯度保护强度",
    "One neighborhood pass per emission round": "每轮发射增加一次邻域分析",
    "Area Light Texture Importance Sampling": "面光纹理重要性采样",
    "Guide photon directions from visible caustics for Area, Sun and zero-radius Point/Spot lights. Area lights use their actual emission positions and textures": "根据可见焦散引导面光、日光和零半径点光／聚光的发射方向。面光使用真实发射位置和纹理",
    "Learn where to emit on Area lights from actual textured emission and visible caustics; retains uniform exploration and supports zero Spread independently of directional guiding": "根据真实纹理发光及可见焦散学习面光上的发射位置，保留均匀探索。可独立于方向引导使用，并支持零扩散角",
    "Guides Area, Sun and zero-radius Point/Spot": "引导面光、日光和零半径点光／聚光",
    "Learns textured emission and visible contribution": "学习纹理发光与可见贡献",
    "Zero Spread uses position guiding only": "零扩散角仅使用位置引导",
    "3D Gaussian Emission Guiding": "3D 高斯发射引导",
    "Photon Guiding": "光子路径引导",
    "Gaussian Guiding": "高斯引导",
    "Guides Sun and zero-radius Point/Spot": "引导日光及零半径点光／聚光",
    "Finite lights keep native emission": "有面积的光源沿用原有发射方式",
    "Photon Sampling": "光子采样",
    "Photon Behavior": "光子行为",
    "Photon Presets": "光子预设",
    "Add Photon Preset": "添加光子预设",
    "High Precision": "高精度",
    "Preview": "预览",
    "Gather Limit": "聚集求值上限",
    "Photon Noise Strength": "光子噪声强度",
    "Local Neural Bandwidth": "轻量神经网络展宽",
    "Use a small pretrained network to refine surface photon width from local coverage, ellipse shape and dispersion": "使用小型预训练网络，根据局部覆盖、椭圆形状和色散微调表面光子的核宽度",
    "Cell Adaptive Neighborhood": "空间单元自适应邻域",
    "Cell statistics and adaptive queries per emission round": "每轮发射执行单元统计与自适应邻域查询",
    "Ellipse Photon Noise Factor": "椭圆光子噪声系数",
    "Rough Photon Spread": "粗糙光子展宽",
    "Caustics": "焦散",
    "Use Photon Cache": "使用光子缓存",
    "Photon Calculation": "光子计算顺序",
    "Simultaneous": "同时",
    "Use Tiling": "使用平铺",
    "Before Sampling": "采样前",
    "After Sampling": "采样后",
    "Deep output uses combined convergence": "Deep 输出沿用联合收敛与常规采样预算",
    'Separate Photon Sampling': '独立光子采样',
    'Photon Sampling Budget': '光子采样预算',
    'Shared Maximum Samples': '共用最大采样数',
    'Independent Maximum Samples': '独立最大采样数',
    'Maximum Photon Samples': '光子最大采样数',

    "Shadow Caustics Method": "阴影焦散方法",
    "Manifold": "流形（MNEE）",
    "Photon": "光子",
    "Photons per Round": "每轮光子数",
    "Photon Radius": "聚集半径",
    "Photon Footprint": "光子覆盖方式",
    "Fixed Radius": "固定半径",
    "Differential Ellipse": "微分椭圆（自适应尺寸）",
    "Minimum Photon Radius": "最小光子半径",
    "Photon Scatter": "光子散布",
    "Ellipse Scatter Factor": "椭圆散布系数",
    "Photon reception is automatic": "光子焦散自动接收",
    "Emission Interval": "发射间隔",
    "Use CPU, one CUDA/OptiX GPU, or identical OptiX GPUs": "请选择 CPU、单张 CUDA/OptiX 显卡或同型号 OptiX 多卡",
    "CPU + OptiX test: each device uses the sample budget": "CPU＋OptiX 测试模式：每台设备使用完整采样预算",
    "Each GPU uses the sample budget; results are averaged": "每张显卡使用完整采样预算，结果按样本数平均",
    "Photon multi-GPU requires identical OptiX GPUs without CPU": "光子多卡渲染要求同型号 OptiX 显卡，不能混用 CPU",
    "Multi-GPU photon rendering in Deep EXR is not supported": "光子多卡渲染暂不支持 Deep EXR",
    "Rolling shutter is not supported": "暂不支持卷帘快门",
    "Sun photon caustics require bounded mesh or VDB volumes": "日光光子焦散需要有边界的网格或 VDB 体积，请勿使用世界体积",
    "OSL is not supported": "暂不支持 OSL",
    "Custom ray visibility is not supported": "暂不支持自定义光线可见性",
    "Shadow catchers are not supported": "暂不支持阴影捕捉器",
    "Light and shadow linking are not supported": "暂不支持光照关联和阴影关联",
    "Shadow linking on caustic lights is not supported": "暂不支持焦散光源本身的阴影关联",
    "CUDA photon caustics may be slow; prefer OptiX": "CUDA 光子焦散性能可能较低，建议优先使用 OptiX",
    "Volumes are not supported": "暂不支持体积物体",
    "Use mesh casters": "请使用网格物体投射焦散",
    "Only Point, Spot, Sun and Area lights are supported": "目前仅支持点光、聚光、日光和面光",
    "Enable Cast Shadow on caustic lights": "请启用焦散光源的投射阴影",
    "Colored caustic shadows are not supported": "暂不支持彩色焦散阴影",
    "Volume shaders are not supported": "暂不支持体积着色器",
    "Enable Shadow Caustics light, caster and receiver": "请启用光源、投射物体和接收物体的阴影焦散",
    "No caustic light or caster: continuing ordinary path tracing":
        "缺少焦散光源或投射物体：继续普通路径追踪",
}

# RNA property and enum tooltips use Blender's default translation context.
_TOOLTIPS_ZH = {
    "Use shutter-time light transforms, rigid mount pivots and corrected shutter curve endpoints for regular paths":
        "普通光路在快门时刻计算移动灯光，使用刚性父级的共同支点，并修正快门曲线的端点采样",
    "Use moving lights, rigid mount pivots and corrected shutter curve endpoints for photon emission and camera gathering":
        "光子发射与相机侧光子查询均使用移动灯光和刚性父级的共同支点，并修正快门曲线的端点采样",
    "Share cell occupancy to adapt the statistical search radius; preserves photon power and path identity. Requires statistical radius multiplier above 1": "共享空间单元占据统计，自适应调整统计查询半径；保留光子功率与路径归属。统计半径乘数大于 1 时生效",
    'Pilot-neighborhood filter preserving photon power and LPE path membership': '重建邻域筛选；保留光子功率与 LPE 路径归属',
    'Use the original local coverage and gradient estimate': '使用原有局部覆盖程度和梯度估计',
    'Also compare the shading normal of the last sampled caster interface': '额外比较最后一次采样的投射界面着色法线',
    'Also require the same ordered object, material and scattering events': '额外要求投射路径的物体、材质与散射事件按顺序一致',
    'Require both matching caster paths and compatible last-interface normals': '同时要求投射路径一致且上一界面法线相容',
    'Maximum angle between last-interface shading normals used in reconstruction statistics': '重建统计中上一界面着色法线之间允许的最大夹角',
    "Expand sparse photon kernels using a local coverage estimate. Independent of emission guiding; adds one neighborhood pass per emission round": "根据局部覆盖程度扩大稀疏光子的覆盖范围，与发射引导独立；每轮发射增加一次邻域分析",
    "Limit adaptive expansion near reliable photon-flux and color gradients; ignores a statistical noise floor": "在可靠的光通量或颜色梯度附近限制展宽，过滤采样噪声造成的虚假梯度",
    "Desired local photon coverage; larger values favor smoother reconstruction": "期望的局部光子覆盖数量，增大有利于平滑稀疏区域",
    "Maximum expansion of the existing circular or elliptical footprint. 1 disables adaptive reconstruction; capped at 4 to bound neighborhood work": "已有圆形或椭圆覆盖范围的最大放大倍率；1 关闭自适应重建，上限 4 用于限制邻域查询开销",
    "How strongly reliable gradients limit radius expansion. 0 is coverage-only reconstruction; does not change photon power": "可靠梯度对半径扩大的限制强度；0 仅按覆盖程度重建，不修改光子功率",
    "Learn photon emission from visible caustics. Currently guides Sun and zero-radius Point/Spot lights; finite lights keep their native proposal": "根据当前相机可见的焦散学习光子发射分布，包含经镜面路径看到的贡献。当前引导日光和零半径点光／聚光，有面积的光源沿用原有发射方式",
    "Render large images in tiles to reduce render buffer memory": "将大图分块渲染以减少渲染缓冲占用",
    "Calculate photons before native lighting": "先计算光子，再计算常规光照",
    "Calculate photons after native lighting": "先计算常规光照，再计算光子",
    "Progress native lighting and photons together": "同时推进常规光照和光子采样；使用平铺时不可用",
    "Calculate the photon image first, then add native render tiles": "先计算并显示整图光子贡献，再叠加常规渲染图块",
    "Calculate native render tiles first, then add the photon image": "先计算常规渲染图块，再计算并逐步显示光子贡献",
    'Sample native lighting and photon caustics independently, then combine their normalized estimates': '分别采样常规光照和光子焦散，再合并各自归一化后的结果',
    'Use the native maximum sample count for both estimators': '常规光照和光子分别使用常规最大采样数作为上限；停止状态相互独立',
    'Set the photon sample limit separately from native lighting': '单独设置光子上限，不跟随常规光照采样数',
    'Maximum camera samples used to estimate photon caustics; emission interval controls map reuse': '估计光子焦散的最大相机采样数；发射间隔决定每张光子图的复用次数',

    "Reconstruct photons with a fixed circular kernel": "使用固定圆形核重建光子密度",
    "Propagate ray differentials through specular surfaces to adapt photon size and shape":
        "沿镜面反射和折射传播光线微分，自适应计算每颗光子的椭圆大小与形状",
    "Base ellipse semi-axis lower bound; 0 uses five percent of Photon Radius, before adaptive reconstruction":
        "椭圆基础半轴下限；0 表示聚集半径的 5%，自适应重建可在此基础上扩大足迹",

    "Random tangent displacement of the photon lookup per camera sample, "
    "as a fraction of Photon Radius. Breaks up fixed photon spots at the cost of "
    "additional noise and smoothing; 0 disables scattering":
        "每次相机采样沿接收面随机偏移光子查询位置，偏移范围为聚集半径的指定比例。"
        "打散固定光子斑点，但会增加噪声和平滑；设为 0 关闭散布",
    "Method for lights and objects with Shadow Caustics enabled":
        "为启用了阴影焦散的光源和物体选择计算方法",
    "Use manifold next-event estimation for shadow caustics":
        "使用流形下一事件估计（MNEE）计算阴影焦散",
    "Use biased photon caustics on CPU, CUDA or OptiX; "
    "unsupported configurations are disabled or reported as errors":
        "在 CPU、CUDA 或 OptiX 上使用有偏光子焦散；"
        "不支持的配置将禁用或报错，不会自动切换方法",
    "Number of light paths attempted in each emission round. "
    "More photons reduce noise at the cost of tracing and gathering time":
        "每轮尝试发射的光路数量。增加光子数可降低噪声，但会增加追踪和聚集耗时",
    "Base world-space radius for photon density estimation before adaptive reconstruction. "
    "Smaller radii preserve detail but require more photons":
        "自适应重建之前的世界空间基础聚集半径。减小半径可保留细节，但需要更多光子",
    "Camera samples between photon emission rounds. "
    "1 emits a new photon map for every sample; larger values reuse each map longer "
    "to reduce emission overhead, but average fewer independent photon maps":
        "两轮光子发射之间的相机采样数。设为 1 时每次采样都发射新光子图；"
        "较大值会更久地复用每张光子图，降低发射开销，但参与平均的独立光子图更少",
}

_ZH = {(CONTEXT, text): translation for text, translation in _LABELS_ZH.items()}
_ZH.update({
    ("*", "Glossy Terminator Smoothing (Disney)"): "高光明暗交界柔化（Disney）",
    ("*", "Smooth bumped glossy reflections at the light terminator while preserving the flake normals"):
        "在光照明暗交界处平滑衰减受凹凸影响的高光反射，保留颗粒法线方向",
})
# RNA tooltips and UI callers without an explicit context use the default lookup.
# Keep the guiding labels and descriptions available through both entry points.
_ZH.update({('*', text): translation for text, translation in _GPU_GUIDING_ZH.items()})
for _label in ("Use Relative Motion Blur", "Correct Photon Motion Blur"):
    _ZH[('*', _label)] = _LABELS_ZH[_label]
_ZH.update({('*', text): translation for text, translation in _HDR_PHOTON_ZH.items()})
_DESCRIPTIONS_ZH = {
    'Method for lights and objects with Shadow Caustics enabled': '用于已启用阴影焦散的灯光和物体的计算方法',
    'Expand sparse photon kernels using a local coverage estimate. Independent of emission guiding; adds one neighborhood pass per emission round': '根据局部覆盖估计展宽稀疏光子核。与发射引导独立，每轮发射后进行邻域重建',
    'Limit adaptive expansion near reliable photon-flux and color gradients; ignores a statistical noise floor': '根据可靠的光子通量与颜色梯度限制自适应展宽，排除统计底噪的影响',
    'Desired local photon coverage; larger values favor smoother reconstruction': '期望的局部光子覆盖数量；较大值倾向于更平滑的重建',
    'Share cell occupancy to adapt the statistical search radius; preserves photon power and path identity. Requires statistical radius multiplier above 1': '共享空间单元占据统计，自适应调整统计查询半径，保留光子功率与路径身份。统计半径乘数必须大于 1',
    'Pilot-neighborhood filter preserving photon power and LPE path membership': '筛选用于重建统计的邻居，保留光子功率与 LPE 路径归属',
    'How strongly reliable gradients limit radius expansion. 0 is coverage-only reconstruction; does not change photon power': '可靠梯度对半径展宽的限制强度。0 表示仅根据覆盖程度重建；不改变光子功率',
    'Sample native lighting and photon caustics independently, then combine their normalized estimates': '分别采样常规光照与光子焦散，再合并各自归一化后的结果',
    'Maximum camera samples used to estimate photon caustics; emission interval controls map reuse': '用于估计光子焦散的最大相机采样数；发射间隔控制光子图的复用',
    'Number of light paths attempted in each emission round. More photons reduce noise at the cost of tracing and gathering time': '每轮发射尝试的光路径数。增加光子可降低噪声，同时增加追踪与聚集耗时',
    'Maximum photon evaluations per surface shading point across all lights. 0 evaluates all candidates; a finite limit adds sampling variance while preserving mean energy': '每个表面着色点对所有灯光的光子求值总上限。0 表示计算全部候选；有限上限会增加采样方差，但保留平均能量',
    'Base ellipse semi-axis lower bound; 0 uses five percent of Photon Radius, before adaptive reconstruction': '自适应重建前的基础椭圆半轴下限；0 表示使用光子半径的 5%',
    'Base world-space radius for photon density estimation before adaptive reconstruction. Smaller radii preserve detail but require more photons': '自适应重建前用于光子密度估计的世界空间基础半径。较小半径保留更多细节，但需要更多光子',
    'Smoothly spread photon spots by jittering the lookup in the surface tangent plane. Measured in photon radii; 0 disables spreading. Use Photon Noise Strength for grains': '在表面切平面内抖动查询位置，平滑扩散光点。以光子半径为单位；0 关闭散布。颗粒由光子噪声强度控制',
    'Independently spread differential photon footprints, including circular fallbacks. Measured in footprint radii; 0 disables spreading. Does not multiply Photon Scatter': '独立控制微分光子足迹的平滑散布，包含圆形回退；以足迹半径为单位，0 关闭散布。此值不与普通光子散布相乘',
    'Blue-noise grains in fixed-radius surface photons and volume photons. 0 keeps smooth kernels; 1 samples full-height grains. Preserves mean energy and converges with samples; does not enlarge photons': '固定半径表面光子与体积光子的蓝噪声颗粒强度。0 保持平滑光子核，1 使用完整幅度的颗粒采样。保留平均能量并随采样收敛；不放大光子',
    'Independent blue-noise grain strength in differential footprint mode, including circular fallbacks. Not a multiplier of Photon Noise Strength': '微分足迹模式的独立蓝噪声颗粒强度，也作用于退回圆形足迹的情况。该值不是光子噪声强度的乘数',
    'Broaden local footprints around sampled BSDF directions. Preserves native blue-noise query scatter; bounded to four times Photon Radius before adaptive reconstruction. 0 disables widening, not rough BSDF transport': '围绕采样的 BSDF 方向展宽局部足迹，保留蓝噪声查询散布。自适应重建前限制在光子半径的四倍以内。0 关闭展宽，但仍计算粗糙材质的光传输',
    'Camera samples between photon emission rounds. 1 emits a new photon map for every sample; larger values reuse each map longer to reduce emission overhead, but average fewer independent photon maps': '两轮光子发射之间的相机采样数。1 表示每次采样都发射新光子图；较大值延长每张图的复用以降低发射开销，但参与平均的独立光子图更少',
    'Photon or MNEE caustics including material color, separated from ordinary light components; Combined remains complete': '包含材质颜色的光子或 MNEE 焦散，从常规光照分量中分离；合成通道仍保留完整画面',
}
for _text, _translation in _DESCRIPTIONS_ZH.items():
    _ZH[(CONTEXT, _text)] = _translation
    _ZH[('*', _text)] = _translation

_REGIONAL_ZH = {
    'Choose each photon multiplier from original footprint overlap; keep its ellipse orientation and aspect ratio, using a bounded Gaussian overlap approximation for statistics': '根据原始足迹的重叠为每个光子选择乘数，保留椭圆方向与长宽比；统计阶段使用有界高斯重叠近似',
    'Local Scalar and Adaptive Ellipse Multiplier: maximum expansion of each original ellipse. Regional modes: support relative to Gather Radius. 1 disables reconstruction': '局部等比例及椭圆自适应乘数：限制每个原始椭圆的放大倍数。区域模式：限制相对聚集半径的范围。1 关闭重建',
    'Normal comparison tolerance; interface continuity compares normals predicted from local derivatives': '法线比较容差；开启上一界面连续性时，比较局部导数预测的法线，而不是直接限制曲面的法线变化',
    'Bandwidth estimates preserving individual photon power and path membership': '核宽度估计；各模式均保留单个光子的功率与路径归属',
    'Scale the original footprint uniformly using its local neighborhood': '根据局部邻域等比例缩放原始光子足迹',
    'Use a separate statistical radius and effective weighted coverage to broaden sparse kernels': '使用独立统计半径与有效加权覆盖，展宽稀疏光子核',
    'Also protect directional gradients and differential structure with independent axis expansion': '进一步利用方向梯度与微分结构保护边缘，分别控制不同方向的展宽',
    'Statistical radius relative to Gather Radius. Local Scalar: 1 keeps footprint-local statistics; larger values extend coverage statistics while keeping gradient protection local. Larger neighborhoods cost more': '统计范围相对聚集半径计算。局部等比例：1 保留足迹局部统计，大于 1 扩大覆盖统计，梯度保护仍使用局部邻域；扩大范围会增加开销',
    'Weight regional neighbors by last-interface position, normal derivatives and scattering identity; adds an optional 48-byte record per emission slot': '根据上一界面的位置、法线导数与散射身份加权邻域；开启后每个发射槽额外使用 48 字节',
    'Legacy: maximum axis expansion. Regional modes: maximum support relative to Gather Radius (rough kernels retain their larger bound). 1 disables reconstruction': '原有模式限制半轴放大倍数；区域模式限制相对聚集半径的最大覆盖范围（粗糙光子保留较大的原有边界）。1 关闭重建',
}
for _text, _translation in _REGIONAL_ZH.items():
    _ZH[(CONTEXT, _text)] = _translation
    _ZH[('*', _text)] = _translation
for _text in ('Reconstruction Method', 'Local Scalar', 'Regional Coverage', 'Adaptive Ellipse Multiplier',
              'Regional Anisotropic', 'Statistical Radius Multiplier', 'Previous Interface Continuity',
              'Compatibility Neighbor Selection', 'Neighbor Evaluations', 'OptiX Photon SER',
              'Local Neural Bandwidth',
              'Use a small pretrained network to refine surface photon width from local coverage, ellipse shape and dispersion',
              'Cell Adaptive Neighborhood',
              'Sample ellipse reconstruction statistics from compatible neighbors with probability compensation. Adds variance; does not discard photon illumination or change LPE membership',
              'Weighted evaluations from up to 32 pilot candidates per photon; sparse neighborhoods are evaluated exactly',
              'Reorder photon material shading on supported OptiX devices. Performance depends on shader diversity and live state; CPU and CUDA are unaffected'):
    _ZH[('*', _text)] = _LABELS_ZH[_text]
# RNA labels and explicit UI labels can use either translation lookup context.
for _label in ('Reconstruction Neighbors', 'Same Light and Receiver', 'Last Interface Normal', 'Full Caster Path', 'Caster Path and Normal', 'Interface Normal Angle',
               "3D Gaussian Emission Guiding", "Area Light Texture Importance Sampling",
               "Photon Presets", "Add Photon Preset", "High Precision", "Preview",
               "Force Zero Roughness for Photons",
               "Use zero roughness for caustic photon reflection, transmission and coat. Enable on the object for all its materials or on the material for every object using it. Disable both to use shader roughness",
               "Ignore material Normal Map and Bump for photon casting, keeping smooth and custom mesh normals and true displacement. Enable on the object for all its materials or on the material for every object using it. Disable both to use shader normals",
               "Force Geometry Normals", "Ignore Material Normals for Photons",
               "Photon Behavior", "Gather Limit", "Reflective Caustics", "Refractive Caustics",
               "Allow this material to reflect shadow caustics on objects with Cast Shadow Caustics enabled",
               "Allow this material to transmit shadow caustics on objects with Cast Shadow Caustics enabled",
               "Guide photon directions from visible caustics for Area, Sun and zero-radius Point/Spot lights. Area lights use their actual emission positions and textures",
               "Learn where to emit on Area lights from actual textured emission and visible caustics; retains uniform exploration and supports zero Spread independently of directional guiding"):
    _ZH[("*", _label)] = _LABELS_ZH[_label]
_ZH.update({
    ("*", "Allow this light instance to emit caustic photons independently of other ray visibility"): "允许此灯光实例发射焦散光子，独立于其他射线可见性",
    ("*", "Allow the detected World sun region to emit caustic photons independently of other ray visibility"): "允许检测到的世界环境太阳区域发射焦散光子，独立于其他射线可见性",
    ("*", "Generate shadow caustics when the object and material allow them. Every instancer in the instance hierarchy must also enable Cast Shadow Caustics. MNEE requires a marked receiver; photons are received globally"): "物体与材质均允许时生成阴影焦散。实例层级中的每个实例对象也必须开启投射阴影焦散。MNEE 需要标记接收物体；光子由全局物体接收",
    ("*", "Photon:"): "光子：",
    ("*", "Maximum photon evaluations per surface shading point across all lights. 0 evaluates all candidates; a finite limit adds sampling variance while preserving mean energy"): "每个表面着色点跨全部灯光的光子求值数量上限。0 表示评估所有候选；有限上限会增加采样方差，但保持平均能量",
    ("*", "Force Geometry Normals"): "强制使用几何法线",
    ("*", "Ignore material Normal Map and Bump for photon casting, keeping mesh smooth and custom normals. True geometry displacement still applies"): "光子投射忽略材质的法线贴图和凹凸，保留网格平滑与自定义法线；真实几何置换仍然生效",
    ("*", "Caustics"): "焦散",
    ("*", "Render Only"): "仅渲染",
    ("NodeTree", "Render Only"): "仅渲染",
    ("*", "Output Mode"): "输出模式",
    ("*", "Shader Cache Directory"): "着色器缓存目录",
    ("*", "Directory for Cycles kernels, CUDA and OptiX disk caches; empty uses a separate per-user QCycles cache. Restart Blender after changing. Explicit cache environment variables take precedence"): "Cycles、CUDA 和 OptiX 的磁盘缓存位置；留空使用当前用户独立的 QCycles 缓存目录。修改后重启 Blender；显式设置的缓存环境变量优先",
    ("*", "Photon"): "光子",
    ("*", "Independently spread differential photon footprints, including circular fallbacks. Measured in footprint radii; 0 disables spreading. Does not multiply Photon Scatter"): "独立控制微分光子足迹的平滑散布，包含圆形回退；以足迹半径为单位，0 关闭散布。此值不与普通光子散布相乘",
    ("*", "Smoothly spread photon spots by jittering the lookup in the surface tangent plane. Measured in photon radii; 0 disables spreading. Use Photon Noise Strength for grains"): "平滑扩散光点：在表面切平面内扰动光子查询位置；以光子半径为单位，0 关闭扩散。颗粒感由光子噪声强度控制",
    ("*", "Rough Photon Spread"): "粗糙光子展宽",
    ("*", "Photon Noise Strength"): "光子噪声强度",
    ("*", "Ellipse Photon Noise Factor"): "椭圆光子噪声系数",
    ("*", "Blue-noise grains in fixed-radius surface photons and volume photons. 0 keeps smooth kernels; 1 samples full-height grains. Preserves mean energy and converges with samples; does not enlarge photons"): "控制圆形表面光子和体积光子内部的蓝噪声颗粒。0 保留平滑光斑，1 采样完整峰值的颗粒；保持平均能量，随采样收敛，不改变光子大小",
    ("*", "Independent blue-noise grain strength in differential footprint mode, including circular fallbacks. Not a multiplier of Photon Noise Strength"): "独立控制微分覆盖模式下表面光子的蓝噪声颗粒，包含圆形回退；此值不与光子噪声强度相乘",
    ("*", "Broaden local footprints around sampled BSDF directions. Preserves native blue-noise query scatter; bounded to four times Photon Radius before adaptive reconstruction. 0 disables widening, not rough BSDF transport"): "对采样方向附近的光子足迹作局部展宽，保留常规蓝噪声查询散布；自适应重建之前的最大半轴不超过光子半径的四倍；0 仅关闭展宽，仍计算粗糙反射和折射",
    ("*", "LPE Channels"): "LPE 通道",
    ("*", "LPE Channels — Cached"): "LPE 通道 · 已缓存",
    ("*", "LPE Channels — Uncached"): "LPE 通道 · 未缓存",
    ("*", "Show resident LPE data without restarting sampling; returning to the sampled channel keeps its result"): "显示常驻 LPE 数据而不重启采样；返回原采样通道时保留其结果",
    ("*", "Render this Render Only output on demand; switching to it restarts sampling"): "按需渲染此“仅渲染”输出；切换到它会重新采样",
    ("*", "LPE Viewer"): "LPE 预览器",
    ("*", "Light data '%s' is shared by '%s' and '%s'. Make it single-user before applying."):
        "灯光数据“%s”由“%s”和“%s”共用。请先将数据设为单用户，再应用。",
    ("*", "LPE Preview Overlay"): "LPE 预览覆盖",
    ("*", "Photon contribution queries can be slower on GPU than CPU."): "有光子时，使用 GPU 查询贡献可能比 CPU 慢。",
    ("*", "Show cached LPE output without restarting viewport sampling"): "读取已缓存的 LPE 输出，不重新开始视口采样",
    ("*", "Show the caustics render pass; switching this native pass restarts sampling"): "显示焦散通道；切换这个常规通道会重新渲染",
    ("*", "Produce this output only in the final render"): "仅在最终渲染中输出此通道，不为它单独保留视口缓存",
    ("*", "Preview this output and refresh it in IPR"): "保留此输出的视口缓存，并在交互渲染中更新",
    ("*", "Legacy caustics selection"): "旧版焦散筛选（兼容读取）",
    ("*", "Caustic paths through the selected Cast Shadow Caustics objects; chain casters in Photon mode, connect a receiver, or output directly for all receivers"): "选择经过投射阴影焦散物体的光路；光子模式可按顺序串接投射物体，也可连接接收物体或直接输出全部接收位置",
    ("*", "Photon or MNEE caustics including material color, separated from ordinary light components; Combined remains complete"): "单独输出包含材质颜色的光子或 MNEE 焦散，不再计入普通光照分量；Combined 保留完整画面",
})
_ZH.update({("*", text): translation for text, translation in _TOOLTIPS_ZH.items()})
_ZH[("*", "Render the photon estimator once for the full image in a compact buffer, then combine with native render tiles")] = "使用专用紧凑缓冲计算整张图的光子贡献，再与常规分块结果合成，避免逐块重复发射光子"
_ZH.update({
    ("*", "Nested Dielectric"): "嵌套电介质",
    ("*", "Merge Same Material"): "融合相同材质",
    ("NodeTree", "Photon Path"): "光子光程",
    ("*", 'Photon'): '光子',
    ("*", 'Photon Path'): '光子光程',
    ("*", 'Is Photon Ray'): '是光子射线',
    ("*", 'Photon Ray Length'): '光子射线长度',
    ("*", 'Photon Depth'): '光子深度',
    ("*", 'Diffuse Length'): '漫射长度',
    ("*", 'Glossy Length'): '光泽长度',
    ("*", 'Reflection Length'): '反射长度',
    ("*", 'Reflection Depth'): '反射深度',
    ("*", 'Transmission Length'): '透射长度',
    ("*", 'Volume Length'): '体积长度',
    ("*", 'Volume Depth'): '体积散射深度',
    ("*", 'Volume Layers'): '经过体积层数',
    ("*", 'Output photon caustics separately for each light group, including surface and volume contributions'): '按灯光组单独输出光子焦散，包含表面和体积贡献',
    ("*", 'Portal Depth'): '传送深度',
    ("*", 'Read light-side photon path information and native camera-ray classification'): '读取光源端光子路径信息以及原生相机射线分类',
    ("*", "Treat overlapping objects sharing this material as one medium. Requires nonzero Medium Priority and Nested Dielectric on every object and instancer"): "将共享此材质的重叠物体视为同一介质。材质优先级须大于零，物体及每层实例均须启用嵌套电介质",
    ("*", "Medium Priority"): "介质优先级",
    ("*", "Disabled (Priority 0)"): "未启用（层级 0）",
    ("*", "Set Medium Priority in material settings"): "在材质设置中指定介质优先级",
    ("*", "Every instancer must opt in; adds medium tracking cost"): "每层实例均须启用；会增加介质跟踪开销",
    ("*", "Enable extra medium tracking for closed dielectrics. Requires a positive material Medium Priority and this switch on every instancer"): "为封闭电介质启用额外介质跟踪。材质优先级须大于零，且每层实例均须开启此开关",
    ("*", "Zero disables nested dielectric handling for this material. Higher values replace lower values in overlaps. The object and every instancer must also enable Nested Dielectric"): "零表示此材质不参与嵌套电介质。重叠区域由较高优先级取代较低优先级。物体及每层实例均须启用嵌套电介质",
    ("*", "Resolve overlapping closed dielectric objects using explicit priorities"): "通过显式优先级处理相互重叠的封闭电介质物体",
    ("*", "Higher priority replaces lower priority in overlaps; use distinct priorities for overlapping media"): "重叠部分由较高优先级介质占据；重叠介质应使用不同优先级",
    ("*", "Photon illumination from this source, including reflected and transmitted views and subsequent indirect light; already included in Total Light"): "此光源的光子照明，包含反射、透射中可见的结果及后续间接光；总灯光中已包含此贡献",
})
for _text, _translation in {
    "Integrator": "积分器",
    "Path Tracing": "路径追踪",
    "Use the Cycles unidirectional path tracer": "使用 Cycles 的单向路径追踪积分器",
    "Vertex Connection and Merging (VCM)": "顶点连接与合并（VCM）",
    "Connect and merge light and camera paths": "连接与合并光源路径和相机路径",
    "Light Paths per Round": "每轮发射光路数",
    "Light paths per round across all emission devices, each paired with a camera path": "每轮由所有发射设备共享的光路总数，每条光源路径配对一条相机路径",
    "Unsupported VCM light paths: %s": "VCM 尚未支持的光路：%s",
    "Denoising input: color": "降噪输入：颜色",
    "VCM viewport denoising requires OptiX or Open Image Denoise": "VCM 视口降噪支持 OptiX 和 Open Image Denoise",
    "VCM LPE: connect Light or World Total Light directly to an output": "VCM 的 LPE 支持将灯光或世界的总灯光直接连接至输出",
    "Final passes are listed under View Layer": "最终渲染支持的通道见视图层设置",
    "Available VCM passes are shown below": "下方列出 VCM 支持的通道",
    "VCM uses a global shutter": "VCM 使用全局快门",
    "VCM accumulates light paths across the full frame": "VCM 在整帧上累积光路贡献",
    "Number of independent light path batches to connect and merge": "用于顶点连接与合并的独立光路批次数",
    "Select an OptiX GPU in Preferences": "请在偏好设置中启用一张 OptiX 显卡",
    "Viewport and final render · Combined pass": "视口和最终渲染 · 合成通道",
    "Rendering with Cycles Path Tracing": "当前使用 Cycles 路径追踪",
    "Volume transport needs the native integrator": "体积光传输需要原生积分器",
    "Subsurface scattering needs the native integrator": "次表面散射需要原生积分器",
    "Ray portals need the native integrator": "光线传送需要原生积分器",
    "The shader uses camera or path state": "着色器依赖相机射线或路径状态",
    "The surface needs the native integrator": "此表面特性需要原生积分器",
    "The light configuration needs the native integrator": "此灯光配置需要原生积分器",
    "The camera projection or depth of field needs the native integrator": "此相机投影或景深需要原生积分器",
    "Static surfaces and a pinhole camera": "支持静态表面和针孔相机",
    "Simple Deep Color": "简化 Deep 颜色",
    "Save each input as a PNG image": "将每个输入分别保存为 PNG 图像",
    "Save lighting samples with depth in a Deep EXR file": "将带深度的光照样本保存到 Deep EXR 文件",
    "Distribute Beauty over depth samples instead of recording physical depth colors. Deep LPE requires physical colors":
        "将完整画面颜色分配到深度样本。关闭后记录实际贡献所属深度的颜色；Deep LPE 始终使用此物理模式",
    "Photon Query Index": "光子查询索引",
    "Group photon queries by light and surface or volume":
        "按灯光及表面／体积分组查询光子",
}.items():
    _ZH[(CONTEXT, _text)] = _translation
    _ZH[('*', _text)] = _translation
for _text, _translation in {'Deep Content': 'Deep 内容', 'Occlusion Deep': '遮挡 Deep', 'Contribution Deep': '贡献 Deep', 'Lighting contribution or complete occlusion': '选择光照贡献样本或完整遮挡信息', "Retain full depth and alpha, including unlit fog and occluding surfaces. Suitable for depth occlusion and inserting objects. Do not composite each light's complete opacity repeatedly": '保留完整深度和 Alpha，包括未被照亮的雾和遮挡表面。适合深度遮挡、插入物体；不要对每盏灯重复叠加同一份完整遮挡。', 'Keep only samples contributing to the selected lighting channels. Usually smaller and lighter. RGB preserves camera-path attenuation; alpha describes retained samples only. Add flattened lighting contributions; do not treat this as full scene occlusion. Use Combined Occlusion Deep when full scene occlusion is needed': '只保留对所选光照通道有贡献的样本，通常体积更小、更轻便。RGB 保留相机路径衰减，Alpha 仅代表保留的样本。压平后以加法合成光照；它不是完整场景遮挡，需要完整遮挡时请另输出 Combined 遮挡 Deep。', 'Deep output requires direct lighting inputs from the same Render Layers source.': 'Deep 输出需要直接连接同一场景、同一视图层的光照通道。', 'Deep supports captured lighting passes only. Z, normals, vectors, Cryptomatte and other data passes are unsupported. Render again after connecting lighting inputs.': 'Deep 仅支持已记录的光照通道；不支持 Z、法线、运动矢量、Cryptomatte 等数据通道。连接光照接口后请重新渲染。', 'Failed to write Deep EXR; see render log.': 'Deep EXR 写入失败，请查看渲染日志。'}.items():
    _ZH[("*", _text)] = _translation
for _text, _translation in {
    "IPR detected. GPU sampling is available in this viewport.": "已检测到 IPR，可在该视口使用 GPU 采样",
    "Query Mode": "查询方式", "Detailed Paths": "详细光路", "Total Light": "总灯光",
    "Group all events on the same ordered object and material route": "合计按相同顺序经过物体和材质的全部事件",
    "Rank each scattering event sequence separately": "按每条散射事件序列分别排名",
    "Contribution Query": "贡献查询", "Sampling Region": "采样区域", "Sampling Results": "采样结果",
    "Clear Region": "清除", "Event Detail": "事件详细程度", "Compact": "精简", "Detailed": "详细",
    "Object Path": "物体光路", "Material Path": "材质光路",
    "Add Object Path": "添加物体光路", "Add Material Path": "添加材质光路",
    "Local Render": "本次局部渲染", "Unfinished": "未完成", "Waiting for view": "等待视角稳定",
    "Over 1024 pixels. CPU evaluation may be slow.": "选区超过 1024 个像素，使用 CPU 评估可能较慢。",
    "Open a rendered 3D View or Render Result to select pixels": "请打开渲染视口或渲染结果并框选像素",
    "Drag a box in the rendered view or Render Result": "请在渲染视口或渲染结果中拖动框选",
    "Missing or ambiguous datablock": "数据块已丢失或同名", "Query result changed": "查询结果已更新",
    "The current LPE tree is read-only": "当前 LPE 节点树只读",
    "LPE state budget exceeded": "超过当前 LPE 节点图的状态上限",
    "Viewport GPU scene is not resident": "视口 GPU 场景尚未就绪",
    "Viewport projection does not match the selection": "视口投影与选区不匹配",
    "Scene or projection requires the CPU query": "当前场景或相机需要 CPU 查询",
    "Insufficient free memory for the GPU query": "GPU 查询可用显存不足",
    "GPU contribution query interrupted": "GPU 查询已中断",
    "Viewport scene changed": "视口场景已更改",
    "Viewport session closed": "视口渲染会话已关闭",
    "Viewport query superseded": "视口查询已被新请求取代",
    "Could not save contribution query scene": "无法保存贡献查询使用的场景副本",
    "No compatible NVIDIA viewport": "没有兼容的 NVIDIA 渲染视口",
    "Estimated extra GPU memory: %.1f MiB": "预计额外显存 %.1f MiB",
    "Query Device": "查询设备", "CPU fallback": "CPU（未使用视口 GPU）",
    "Use a compatible NVIDIA viewport, otherwise use CPU": "有兼容的 NVIDIA 渲染视口时使用 GPU，其余情况使用 CPU",
    "Start Contribution Query": "开始贡献查询", "Resume": "继续", "Paused": "已暂停",
    "Query Samples": "采样次数", "Photon Samples": "光子采样", "Completed": "已完成",
    "Sampling": "采样中", "CPU query did not complete": "CPU 查询未完成",
}.items():
    _ZH[("*", _text)] = _translation
    _ZH[("Operator", _text)] = _translation
translations_dict = {"zh_HANS": _ZH, "zh_CN": _ZH}


def register():
    bpy.app.translations.register(__name__, translations_dict)


def unregister():
    bpy.app.translations.unregister(__name__)
