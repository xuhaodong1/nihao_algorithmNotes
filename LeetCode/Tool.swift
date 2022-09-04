//
//  Tool.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/4.
//

import Cocoa

class BaseCode: NSObject {
    private let classStr: String
    var excuteable: Bool { return false }
    required init(classStr: String) {
        self.classStr = classStr
    }

    func executeTestCode() {
        guard excuteable else { return }
        print(classStr + ":")
    }
}

func registerBaseCode() -> [BaseCode] {
    var cout = UInt32(0)
    guard let imageName = class_getImageName(Solution.self),
          let classes = objc_copyClassNamesForImage(imageName, &cout) else { return [] }
    var baseCodes = [BaseCode]()
    for i in 0 ..< Int(cout) {
        if let clsName = String(cString: classes[i], encoding: .utf8)?.components(separatedBy: ".").last,
           clsName != "BaseCode",
           let cls = swiftClassFromString(clsName) as? BaseCode.Type {
            let baseCode = cls.init(classStr: clsName)
            baseCodes.append(baseCode)
        }
    }
    return baseCodes
}

func swiftClassFromString(_ className: String) -> AnyClass? {
    return NSClassFromString("LeetCode" + "." + className)
}
