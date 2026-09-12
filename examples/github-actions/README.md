# 可选 GitHub Actions 验证

[validate.yml](validate.yml)运行无依赖计算测试、模型导出和归档数值回归。将其复制到仓库 `.github/workflows/validate.yml` 即可启用。

模板当前未启用，不代表远端 CI 已运行。提交 workflow 需要 GitHub 账号或凭据具有相应权限；普通复算直接执行 `npm test` 和 `python3 scripts/verify-archive.py`。
