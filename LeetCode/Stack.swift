//
//  Stack.swift
//  LeetCode
//
//  Created by haodong xu on 2022/11/6.
//

import Foundation

/// 栈等相关练习题
class Stack: BaseCode {

    /// 题目链接：[1106. 解析布尔表达式](https://leetcode.cn/problems/parsing-a-boolean-expression/description/)
    func parseBoolExpr(_ expression: String) -> Bool {
        let eChars = [Character](expression)
        var stack = [String]()
        for char in eChars {
            if char == "!" || char == "&" || char == "|" || char == "t" || char == "f" {
                stack.append("\(char)")
            } else if char == ")" {
                var arr = [String]()
                while stack.last != "!" && stack.last != "&" && stack.last != "|" { arr.append(stack.removeLast()) }
                let last = stack.removeLast()
                if last == "!" { stack.append(arr.first! == "t" ? "f" : "t") }
                else if last == "&" { stack.append(arr.contains("f") ? "f" : "t") }
                else if last == "|" { stack.append(arr.contains("t") ? "t" : "f") }
            }
        }
        return stack.first == "t"
    }
}
