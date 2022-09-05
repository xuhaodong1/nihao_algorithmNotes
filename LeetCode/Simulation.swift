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

//    override var excuteable: Bool {
//        return true
//    }

    override func executeTestCode() {
        super.executeTestCode()
        print(numSpecial([[1,0,0],
                          [0,0,1],
                          [1,0,0]]))
    }
}
