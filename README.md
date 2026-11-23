# Blender QCycles

[English](#english) · [简体中文](#简体中文)

## English

QCycles is a custom build of **Blender 5.3.0** for artists who need to render caustics, identify which light path produced a highlight, and deliver those contributions to compositing. It adds a photon caustics estimator, a Light Path Expressions (LPE) node editor, lightweight GPU surface path guiding, and depth-aware lighting passes to Cycles. The default integrator is Cycles Path Tracing.

**Device support.** This branch supports **NVIDIA GPUs only** for GPU rendering. Use **OptiX** on an RTX card, or **CUDA** where appropriate. CPU rendering remains available, including photon caustics, CPU contribution queries, and Cycles’ OpenPGL guiding in builds that include it. The lightweight GPU surface guide runs on OptiX and CUDA. AMD, Intel and Apple GPU backends are outside this release’s support scope.

### Photon caustics

- **Set up a shot.** In Render Properties → Light Paths → Caustics, choose **Photon**. Enable shadow caustics on the light and the casting object, then enable reflective or refractive caustics on its material. Enable casting on each parent instancer as well. A diffuse floor can receive photon caustics without a receiver flag. The resulting light appears in Combined and can also be isolated in the Caustics pass, LPEs and light groups.
- **Shape the footprint.** Differential ellipses transport a small beam through reflective and refractive interfaces. At the receiver the footprint can be narrow across a bright line and long along it. Fixed-radius footprints are also available. Adaptive reconstruction uses nearby photon coverage and optional gradient protection to smooth sparse deposits. The statistical radius controls how far reconstruction looks for evidence. The maximum radius multiplier controls how far a footprint can expand.
- **Spend photons where they matter.** The 3D Gaussian emission guide learns which caster regions contributed to visible caustics. A separate importance sampler can learn where to emit on a textured area light. An HDR world can emit from a detected sun region with adjustable contrast and angular radius. Individual lights have photon visibility, strength and size controls. Volume photons can reveal a focused beam passing through fog.
- **Control the workload.** Photons per Round and Emission Interval control map density and reuse. A separate sample budget and a compact full-frame cache are available for working with large tiled renders. Before/after sampling controls when caustics appear in the render window. The Photon Path shader node exposes light-side path information such as length, bounce depth and ray-portal depth.

### Light paths and interactive lighting

The **LPE Node Editor** describes contributions by source, ordered objects or materials, and scattering events. For example, isolate light from a single spot that reflects off a metal panel before illuminating a wall. A light or world’s Photon socket selects that source’s photon contribution. Total Light includes eligible caustics. Light groups can also expose individual lighting components, including photons.

Use **Ctrl-Shift-click** to connect a viewer. Outputs set to **Always** and cached Light Edit inputs can be inspected as overlays while interactive rendering continues. **Contribution Query**, in the LPE or Image Editor sidebar, progressively ranks paths for a selected image region. It defaults to CPU and can use a compatible running NVIDIA rendered viewport. A result can be added to the node tree as either an object path or a material path.

### Lightweight GPU surface path guiding

For a room illuminated through a small opening, enable **Render Properties → Sampling → GPU Surface Path Guiding**. It learns productive bounce directions on diffuse and eligible rough reflective surfaces. Start with **128 training samples** and a **0.5 guiding probability**. With a 1,024-sample render budget, the first 128 samples contribute to the image while training, then the guide is frozen for the remaining 896. Compare noise and time in the same indirectly lit crop. Glass transmission, ideal mirrors, hair and volume scattering keep their usual sampling. Photon emission has its own independent 3D Gaussian guide.

### Materials and motion

- **Nested dielectrics.** Enable participation on the object and its instancers, then assign nonzero material priorities. For overlapping glass, ice and water, priorities of 3, 2 and 1 give glass precedence over ice and ice over water. Optional same-material union treats overlapping regions of the same participating material as one medium.
- **Rough reflections.** The material’s Disney glossy terminator softening attenuates abrupt highlights on rough reflective lobes near the light–dark boundary, useful for flaked car paint under grazing light. Camera and photon shading honor it. Object or material overrides can separately force mesh normals or zero roughness for photons.
- **Motion blur.** Ordinary lighting and photons have separate motion-correction switches. Relative correction is useful for a moving rig such as a headlight attached to a vehicle. The photon correction can remain enabled while ordinary lighting uses the original Cycles behavior.

### Deep EXR and overscan

The compositor’s **File Output → Deep EXR** accepts supported lighting passes directly from the same Render Layers node, including caustics, light groups and LPEs. **Contribution Deep** stores depths where the selected pass contributes. For a light shaft, this can isolate the illuminated fog. **Occlusion Deep** also retains unlit occluders for placing other elements at depth. One File Output node can group multiple connected lighting channels in one file. Normals, standalone Z, motion vectors and Cryptomatte are outside the supported Deep lighting passes.

**Output Properties → Overscan** adds rendered pixels outside the delivery frame of an EXR. Set a percentage based on the longer image dimension or specify each edge in pixels. On a 1920 × 1080 frame, 5% adds 96 pixels to each edge. The EXR keeps a 1920 × 1080 `displayWindow` and carries the 2112 × 1272 image in its `dataWindow`. Use a perspective or orthographic camera, single-view output and a disabled Render Region. The camera-view guide previews the margin. Interactive rendering keeps its original framing.

### Getting started and project status

Use the **QCycles build of Blender** to access these renderer features. Enable **QCycles Production Guide** in Preferences → Add-ons, then open **3D Viewport → N → QCycles**. The guide edits the scene’s actual settings and includes per-control help in English and Simplified Chinese. **Vertex Connection and Merging (VCM)** is an incomplete prototype and a future development area. Use Path Tracing for the workflows described above.

Blender as a whole is licensed under the GNU General Public License, version 3. Individual files may carry compatible licenses. [Blender](https://www.blender.org/) · [License](https://www.blender.org/about/license/) · [Blender build instructions](https://developer.blender.org/docs/handbook/building_blender/)

## 简体中文

QCycles 是基于 **Blender 5.3.0** 的定制构建，面向需要渲染焦散、定位高光来自哪条光路，并将光照分层交给合成的制作人员。它为 Cycles 增加了光子焦散估计、光路表达式（LPE）节点编辑器、轻量 GPU 表面路径引导，以及带深度的光照通道。默认积分器为 Cycles 路径追踪。

**设备支持。** 此分支的 GPU 渲染**仅支持 NVIDIA 显卡**。RTX 显卡可选 **OptiX**，需要时可用 **CUDA**。CPU 渲染仍然可用，包括光子焦散、CPU 贡献查询，以及在编入 OpenPGL 时使用 Cycles 的 CPU 路径引导。轻量 GPU 表面引导支持 OptiX 和 CUDA。AMD、Intel 和 Apple 的 GPU 后端不在本版支持范围内。

### 光子焦散

- **搭建焦散光路。** 在「渲染属性 → 光程 → 焦散」中选择**光子**，给灯光和投射物体开启阴影焦散，再在投射物体的材质上允许反射或折射焦散。若物体来自实例，各级实例也要允许投射。漫反射地面可以直接接收光子焦散。焦散会进入 Combined 完整画面，也可从独立焦散通道、LPE 或灯光组中取出。
- **控制光斑及重建。** 微分椭圆沿反射和折射界面传播一束光的宽度和方向，因此落在接收面上的光斑能沿亮线拉长，同时保留狭窄的横向细节。也可选择固定半径。自适应重建依据周围的光子覆盖调整光斑，可结合梯度保护平滑稀疏区域。统计半径决定向多远的邻域取样。最大半径乘数决定单个光斑最多可以扩大多少。
- **提高有效发射率。** 3D 高斯发射引导学习哪些投射区域对可见焦散有贡献。带纹理的面光还可独立学习从灯面哪里发射。HDR 环境支持识别太阳亮区，可调整亮度对比阈值和角半径。每盏灯都有独立的光子可见性、强度和大小设置。场景中有雾时，体积光子可表现聚焦光束穿过介质的路径。
- **分配计算量。** 每轮光子数和发射间隔控制每张光子图的密度与复用次数。大图平铺渲染可使用独立采样预算和紧凑的整图光子缓存。选择采样前或采样后，可以控制何时在渲染窗口看到焦散。着色器中的「光子光程」节点提供光源侧的路程、反弹深度和光线传送深度等信息。

### LPE 与交互调光

**LPE 节点编辑器**按光源、依次经过的物体或材质，以及散射事件筛选光照。例如，可以单独提取「某盏聚光灯经金属板反射后照到墙面」的贡献。灯光和世界环境节点的光子接口可提取该来源的光子贡献。总灯光包含满足条件的焦散。灯光组也能输出各自的光照分量，包括光子。

用 **Ctrl-Shift-左键**连接预览器。设为「总是」的输出和已缓存的灯光编辑输入可作为视口覆盖预览，交互渲染继续累积。「LPE 编辑器」或「图像编辑器」侧栏中的**贡献查询**可以框选一小块画面，渐进排列贡献最大的光路。查询默认使用 CPU，有兼容的 NVIDIA 渲染视口时也可使用 GPU。选中的结果可以按物体光路或材质光路添加回节点树。

### 轻量 GPU 表面路径引导

例如房间主要靠一个小门洞进光，可开启「**渲染属性 → 采样 → GPU 表面路径引导**」。它在漫射面及受支持的粗糙反射面学习更容易带回光的反弹方向。可从**训练采样 128、引导概率 0.5** 开始。总预算设为 1024 时，前 128 个样本同时训练和成图，后 896 个使用固定的引导分布。固定同一块间接照明区域，比较噪声和耗时即可判断收益。玻璃透射、理想镜面、毛发和体积散射沿用常规采样。光子发射另有独立的 3D 高斯引导。

### 材质与运动

- **嵌套电介质。** 在物体和各级实例上开启参与，再给材质设非零优先级。玻璃、冰、水彼此穿插时，可分别设为 3、2、1，使重叠处的玻璃优先于冰，冰优先于水。同一参与材质的区域如需相溶，可另行开启融合。
- **粗糙反射。** 材质中的 Disney 光泽明暗交界柔化可以缓解粗糙反射高光在明暗边界突然断开的问题，适合掠射光下的颗粒车漆。相机与光子着色都使用这项设置。物体和材质还可为光子分别强制使用网格法线或零粗糙度。
- **运动模糊。** 常规光照与光子各有独立的运动修正开关。车灯与车身绑定、共同高速移动时，可使用相对运动修正。也可以只开启光子修正，让常规光照保留原有的 Cycles 行为。

### Deep EXR 与过扫描

合成器的「**文件输出 → Deep EXR**」可以直接接收同一渲染层节点的光照通道，包括焦散、灯光组和 LPE。**贡献 Deep** 保存所选通道实际有贡献的深度样本，例如单独输出雾中被光柱照亮的部分。**遮挡 Deep** 还保留未被照亮的遮挡物，适合按深度插入其他合成元素。同一个文件输出节点可将多个光照接口写入一份文件。法线、独立 Z 通道、运动矢量和 Cryptomatte 不属于受支持的 Deep 光照通道。

「**输出属性 → 过扫描**」为 EXR 在交付画框外多渲染一圈像素。可以按画面长边的百分比统一扩展，也可分别指定四边的像素数。1920 × 1080 的画面设为 5% 时，每边增加 96 像素。EXR 的 `displayWindow` 仍为 1920 × 1080，`dataWindow` 包含 2112 × 1272 的画面数据。使用透视或正交相机、单视图输出，并关闭渲染区域。相机视图的外框会标出边距，交互渲染保持原构图。

### 使用入口与开发状态

以上功能需要运行 **QCycles 版 Blender**。在「偏好设置 → 插件」启用 **QCycles制作指南**，随后打开「**3D 视图 → N → QCycles**」。指南直接读写场景设置，参数说明会随界面语言切换中英文。**VCM（顶点连接与合并）**目前是完成度较低的原型，属于后续开发方向。上述制作流程请使用路径追踪积分器。

Blender 整体遵循 GNU 通用公共许可证第 3 版，个别文件可采用与之兼容的许可证。[Blender 官网](https://www.blender.org/) · [许可说明](https://www.blender.org/about/license/) · [Blender 构建说明](https://developer.blender.org/docs/handbook/building_blender/)
