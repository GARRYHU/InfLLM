# 绕过缓存，强制加载所有 block
force_load = False

# 不加载 block（需修改配置文件中的 max_cached_block，保证缓存能够容纳所有 block）
force_no_load = False

# 输出 block 选择日志（不包括 init tokens、local tokens）
output_hit_log = False

# 输出 miss rate：cpu->gpu transfer: num_transferred/num_selected
output_miss_rate = False

# 输出 block 加载时间: load cache: ms
output_load_time = False
