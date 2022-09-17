//
//  Simulation.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/4.
//

import Foundation

/// 模拟相关练习题
class Simulation: BaseCode {

    /// 题目链接：[1582. 二进制矩阵中的特殊位置](https://leetcode.cn/problems/special-positions-in-a-binary-matrix/)
    func numSpecial(_ mat: [[Int]]) -> Int {
        let n = mat.count, m = mat[0].count
        var cols = [Int](repeating: 0, count: n)
        var rows = [Int](repeating: 0, count: m)
        var ans = 0
        for i in 0..<n {
            for j in 0..<m where mat[i][j] == 1 {
                cols[i] += 1; rows[j] += 1
            }
        }
        for i in 0..<n {
            for j in 0..<m where mat[i][j] == 1 && cols[i] == 1 && rows[j] == 1 {
                ans += 1
                break
            }
        }
        return ans
    }

    /// 题目链接：[1598. 文件夹操作日志搜集器](https://leetcode.cn/problems/crawler-log-folder/)
    func minOperations(_ logs: [String]) -> Int {
        return logs.reduce(0) { partialResult, log in
            if log == "../" {
                return max(partialResult - 1, 0)
            } else if log == "./" {
                return partialResult
            } else {
                return partialResult + 1
            }
        }
    }

    /// 题目链接：[1619. 删除某些元素后的数组均值](https://leetcode.cn/problems/mean-of-array-after-removing-some-elements/)
    func trimMean(_ arr: [Int]) -> Double {
        let n = arr.count, removeCnt = Int(Double(n) * 0.05)
        return Double(arr.sorted()[removeCnt..<n-removeCnt].reduce(0, +)) / Double(n - removeCnt * 2)
    }

    /// 题目链接：[1624. 两个相同字符之间的最长子字符串](https://leetcode.cn/problems/largest-substring-between-two-equal-characters/)
    func maxLengthBetweenEqualCharacters(_ s: String) -> Int {
        var map = [Character: Int](), ans = -1
        for (index, char) in s.enumerated() {
            if let l = map[char] {
                ans = max(index - l - 1, ans)
            } else {
                map[char] = index
            }
        }
        return ans
    }

//    override var excuteable: Bool {
//        return true
//    }

    override func executeTestCode() {
        super.executeTestCode()
        print(maxLengthBetweenEqualCharacters("mgntdygtxrvxjnwksqhxuxtrv"))
    }
}
