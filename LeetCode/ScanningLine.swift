//
//  ScanningLine.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/16.
//

import Foundation

/// 扫描线等相关练习题
class ScanningLine: BaseCode {

    /// 题目链接：[850. 矩形面积 II](https://leetcode.cn/problems/rectangle-area-ii/)
    func rectangleArea(_ rectangles: [[Int]]) -> Int {
        let MOD = Int(1e9) + 7
        let list = rectangles.flatMap{ [$0[0], $0[2]] }.sorted()
        let n = list.count
        var ans = 0
        for i in 1..<n {
            let l = list[i - 1], r = list[i], len = r - l
            if len == 0 { continue }
            var lines = [[Int]]()
            /// 包含在竖直线段区间内的
            for rectangle in rectangles where rectangle[0] <= l && rectangle[2] >= r {
                lines.append([rectangle[1], rectangle[3]])
            }
            lines.sort { line1, line2 in
                return line1[0] != line2[0] ? line1[0] < line2[0] : line1[1] < line2[1]
            }
            var tot = 0, top = -1
            /// 求并集
            for line in lines where line[1] > top {
                tot += (line[1] - max(line[0], top))
                top = line[1]
            }
            ans += (tot * len)
            ans %= MOD
        }
        return ans
    }

//    override var excuteable: Bool {
//        return true
//    }

    override func executeTestCode() {
        super.executeTestCode()
        print(rectangleArea([[0,0,3,3],[2,0,5,3],[1,1,4,4]]))
    }
}
