# 客户端插件:二次开发调试加载指南

> 适用场景:把**本地修改过的** `slicer_plugin/` 加载进 3D Slicer 进行调试。
> 不涉及打包成官方扩展(那条路径需要 CMake + Slicer 构建环境,本指南不覆盖)。

## 1. 前置认知

- 客户端是 **scripted(纯 Python)的 Slicer 扩展**:核心是
  `SlicerNNInteractive.py` + `.ui` 界面 + 资源文件。**改完即生效,没有编译步骤。**
- 仓库里的 `CMakeLists.txt`(根目录、`slicer_plugin/`、`slicer_plugin/SlicerNNInteractive/`)
  只在打包成官方扩展、提交 Slicer Extensions Index 时才用到。**二次开发完全用不到,
  可以忽略。** 所以这里的"构建"对开发者而言就是"让 Slicer 找到并加载这个目录"。
- 客户端必须配合 server 使用。server 的安装与启动见仓库根 `README.md` 和 `server/`。

## 2. 前置条件

- 已安装最新版 3D Slicer(https://download.slicer.org/)。
- 机器能联网 —— 首次加载模块时会自动用 pip 安装 Python 依赖(见第 4 节)。
- 已拿到本仓库代码(当前工作副本即可,无需额外处理)。

## 3. 两种加载方式

两种方式选择的目录层级**不同**,这是最容易出错的地方,务必看清:

```
仓库根/
  CMakeLists.txt
  slicer_plugin/                      <-- 方式 A(Extension Wizard)选这一层
    CMakeLists.txt
    SlicerNNInteractive/              <-- 方式 B(Additional module paths)选这一层
      SlicerNNInteractive.py
      CMakeLists.txt
      Resources/ ...
```

### 方式 A:Extension Wizard(临时,仅当前会话有效)

1. 打开 Slicer,模块下拉菜单 > `Developer Tools` > `Extension Wizard`。
2. 点 `Select Extension`。
3. 选择仓库内的 **`slicer_plugin/` 目录**(该目录直接含 `CMakeLists.txt` 和
   `SlicerNNInteractive/` 子目录)。
4. Slicer 会立即加载模块。

特点:简单快速,但**关闭 Slicer 后失效**,下次要重新选。

### 方式 B:Additional module paths(持久化,二次开发推荐)

1. `Edit` > `Application Settings` > `Modules`。
2. 在 `Additional module paths` 中点 `Add`,选择
   **`slicer_plugin/SlicerNNInteractive/` 目录**(即直接含 `SlicerNNInteractive.py`
   的那一层)。
3. 重启 Slicer。之后每次启动都会自动加载该模块。

特点:配置一次,长期有效,适合反复迭代。

> 对比记忆:**Wizard 选父目录 `slicer_plugin/`;module path 选子目录
> `slicer_plugin/SlicerNNInteractive/`。** 选错层级会导致模块加载不出来。

## 4. 首次加载行为

模块 `setup()` 启动时会调用 `install_dependencies()`,自动用 `pip` 安装三个依赖:

- `requests_toolbelt` —— 分块 HTTP 上传(lasso/scribble mask)。
- `scikit-image` —— lasso 多边形栅格化等图像处理。
- `matplotlib` —— 测试中的 lasso mask 构造等。

安装期间会弹出进度条。**断网会导致安装失败**,模块随之报错。装过一次后后续加载会跳过。

## 5. 打开并配置模块

1. 模块下拉菜单 > `Segmentation` 分类 > 选择 **`nnInteractive`**。
   - 注意:菜单里显示的标题是 **`nnInteractive`**,与目录名
     `SlicerNNInteractive` 不一致,别按目录名去找。
2. (可选)`Edit` > `Application Settings` > `Modules`,把 `nnInteractive`
   从 `Modules` 列表拖进 `Favorite Modules`,之后可在顶部工具栏快速进入。
3. 切到模块的 `Configuration` 标签页,填入 server 的 URL
   (形如 `http://localhost:1527`,**必须带 `http://`**),点 `Test Connection` 验证。

## 6. 二次开发迭代:热重载

改完代码后无需重启 Slicer,用热重载即可:

1. `Edit` > `Application Settings` > `Developer`,勾选 `Enable developer mode`。
   开启后,模块面板顶部会出现 `Reload` 与 `Reload and Test` 按钮。
2. 修改 `SlicerNNInteractive.py` 或 `Resources/UI/SlicerNNInteractive.ui` 后:
   - 点 `Reload` —— 重新导入模块、重建界面。
   - 点 `Reload and Test` —— 重载并运行回归测试(需 server 正在运行;详见
     `README.md` 的 Testing 章节)。

热重载的局限(改动较大或行为异常时,直接重启 Slicer 更可靠):

- observer、键盘快捷键、隐藏的 scribble 节点等不一定被完全清理。
- server 端是有状态的单会话;`README.md` 已记录"重置 server 后插件偶发静默失败,
  重载插件或重启 Slicer 通常可解决"。

## 7. 常见问题排查

| 现象 | 排查 |
|------|------|
| 模块列表里找不到 `nnInteractive` | 确认加载的目录层级选对(见第 3 节);打开 Python Console 看加载报错 |
| 首次加载报依赖错误 | 检查网络/代理,重新打开模块触发重装 |
| 改了 `.ui` 界面没变化 | 点 `Reload`;仍不行就重启 Slicer |
| 连不上 server | 核对 `Configuration` 标签页 URL,需带 `http://`;确认 server 进程在跑 |

## 8. 关联文档

- `dev_docs/capabilities_and_interfaces.md` —— 客户端/server 接口与方法梳理。
- `README.md` —— server 安装启动、官方扩展安装、测试说明。
