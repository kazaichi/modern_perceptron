using Images, ImageView
# シグモイド関数
function sigmoid(x)
    return 1.0 ./ (1.0 .+ exp.(-x))
end

# 行列の連結関数
function combine_one(x)
    return hcat(ones(length(x[:,1]), 1), x)
end

# 重みの自動生成関数
function weight(x)
    a = [floor(rand(), 5)]
    for i in 1:(x-1)
        a = hcat(a, floor(rand(), 5))
    end
    return a
end

# 初期化
E = 1
h = 0

# TIME
X = [0 0 0 0 0 1 0 0 0 1 0 1 1 1 0 0 0 0 0 0
    1 1 0 1 1 1 1 0 1 1 0 0 1 0 0 0 1 1 1 1
    1 1 0 1 1 1 1 0 1 1 0 1 0 1 0 0 0 0 0 0
    1 1 0 1 1 1 1 0 1 1 0 1 1 1 0 0 1 1 1 1
    1 1 0 1 1 1 0 0 0 1 0 1 1 1 0 0 0 0 0 0]
imshow(X)

# 入出力の定義
n = size(X)[2]      # 入力素子の数(Xの列数)
l = n               # 出力素子の数
m = 2               # 隠れ素子の数
d = X               # 目標関数
capital_delta = 0.1 # 時間刻み幅
ipsilon = 0.1       # 学習終了条件

# 回数
count = 0
w = []
v = []
# 重みの初期設定
for i in 1:n+1
    w = vcat(w, weight(m))
end
for i in 1:m+1
    v = vcat(v, weight(l))
end

# 学習
while true
    count += 1
    # 隠れ素子の出力を求める。
    f = sigmoid(combine_one(X) * w)
    # 出力素子の出力を求める。
    h = sigmoid(combine_one(f) * v)

    E = 1 / (2 * length(X[:,1])) * sum(sum((h-d).^2, 2), 1)
    if E[1,1] <= ipsilon
        break
    end

    # 修正誤差を求める。
    delta2 = (h-d) .* h .* (1 - h)
    delta1 = (delta2 * transpose(v[2:length(v[:,1]),:])) .* f .* (1-f)

    # 重みの修正最大量 δmax を求める。
    delta_max = 0
    delta_v = transpose(combine_one(f)) * delta2
    for i in 1:length(delta_v[:,1])
        for j in 1:length(delta_v[1,:])
            if abs(delta_v[i,j]) > delta_max
                delta_max = abs(delta_v[i,j])
            end
        end
    end
    delta_w = transpose(combine_one(X)) * delta1
    for i in 1:length(delta_w[:,1])
        for j in 1:length(delta_w[1,:])
            if delta_w[i,j]>delta_max
                delta_max = delta_w[i,j]
            end
        end
    end

    delta = capital_delta / delta_max

    # 重みの更新
    v = v - delta .* delta_v
    w = w - delta .* delta_w
    if count % 50 == 0
        println("$count 回目")
	println(E[1,1])
	imshow(floor.(h, 3))
    end
end

# 最終結果
println("$count 回目")
println(E[1,1])
imshow(floor.(h, 3))
