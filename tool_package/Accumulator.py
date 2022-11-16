
# 实现累加器

class Accumulator():
    """
        for accumulating sums over 'n' variables. 
        累加n个变量的值. 实现的是加和, 并不是存储每次的值, 然后绘图.

    """
    def __init__(self, n):
        """
        初始化, 有几个二维列表. 
        """
        self.data = [0.0] * n # 初始化 # [0.0, 0.0, ...n个]

    
    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]
    
    def reset(self): # 重置
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx): # 魔术方法
        return self.data[idx] # 将对象按照索引取值时, 就调用该函数.  



if __name__ == "__main__":
    accumulator = Accumulator(3)  # 创建实例. 
    print(accumulator.data)

    for i in range(10): # 累计10论
        accumulator.add(i, i, 1) # 

    # 返回损失和精度:
    print(accumulator[0]/accumulator[-1], accumulator[1]/accumulator[-1])

    






