using DataFrames
using Infiltrator
function my_function(x)
    @infiltrate
    y = x * 2  # 여기에 breakpoint
    z = y + 10
    return z
end

# 이 부분이 중요합니다!
result = my_function(5)
println(result)