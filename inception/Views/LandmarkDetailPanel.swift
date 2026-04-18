//
//  LandmarkDetailPanel.swift
//  inception
//

import SwiftUI

struct LandmarkCalloutView: View {
    let landmark: Landmark
    let distanceMeters: Float
    let isLoading: Bool
    let onConfirm: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        
        HStack(spacing: 12) {
            VStack(spacing: 12) {
                Text("Navigate to the \(landmark.className.lowercased())")
                    .font(.system(.headline, design: .rounded, weight: .semibold))
                    .foregroundStyle(.white)
                ZStack {
                    RoundedRectangle(cornerRadius: 14)
                        .fill(accentColor(for: landmark.className).opacity(0.92))
                        .frame(width: 80, height:80 )
                    
                    Image(systemName: symbolName(for: landmark.className))
                        .font(.system(size: 24, weight: .semibold))
                        .foregroundStyle(.white)
                }
            }.frame(maxWidth: .infinity)
            
            Spacer()
            
            VStack(spacing: 16) {
                if isLoading {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(1.05)
                        .frame(width: 28, height: 28)
                } else {
                    Button(action: onConfirm) {
                        Text("GO")
                            .lineLimit(1)
                            .font(.system(.subheadline, weight: .semibold))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 9)
                            .frame(width: 60, height: 60)
                            .background(Color(red: 0.16, green: 0.53, blue: 0.98))
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                            
                    }
                }
                
                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(.white.opacity(0.82))
                        .frame(width: 60, height: 60)
                        .background(.red)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                .disabled(isLoading)
                .opacity(isLoading ? 0.45 : 1.0)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(width: 280)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
        .overlay(alignment: .leading) {
            CalloutPointer()
                .fill(Color.white.opacity(0.18))
                .frame(width: 18, height: 24)
                .offset(x: -8)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 20)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        }
        .shadow(color: .black.opacity(0.28), radius: 18, y: 10)
    }

    private var distanceString: String {
        distanceMeters < 10
            ? String(format: "%.1f 公尺", distanceMeters)
            : String(format: "%.0f 公尺", distanceMeters)
    }

    private func symbolName(for className: String) -> String {
        switch className {
        case "person": return "figure.walk"
        case "car", "truck", "bus": return "car.fill"
        case "motorcycle", "bicycle": return "bicycle"
        case "tv": return "tv.fill"
        case "laptop": return "laptopcomputer"
        case "cell phone": return "iphone"
        case "chair": return "chair.fill"
        case "couch": return "sofa.fill"
        case "bed": return "bed.double.fill"
        case "dining table": return "table.furniture.fill"
        case "potted plant": return "leaf.fill"
        case "bottle", "cup", "wine glass": return "cup.and.saucer.fill"
        default: return "shippingbox.fill"
        }
    }

    private func accentColor(for className: String) -> Color {
        switch className {
        case "person":
            return Color(red: 1.0, green: 0.8, blue: 0.2)
        case "car", "truck", "bus", "motorcycle", "bicycle":
            return Color(red: 0.92, green: 0.28, blue: 0.24)
        case "tv", "laptop", "cell phone", "remote":
            return Color(red: 0.37, green: 0.58, blue: 0.98)
        case "chair", "couch", "bed":
            return Color(red: 0.31, green: 0.72, blue: 0.49)
        case "dining table":
            return Color(red: 0.64, green: 0.48, blue: 0.33)
        default:
            return Color(red: 0.42, green: 0.68, blue: 0.95)
        }
    }
}

struct NavigationSidePromptView: View {
    let distanceMeters: Float
    let isUpdating: Bool
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                ZStack {
                    Circle()
                        .fill(Color.orange)
                        .frame(width: 40, height: 40)

                    Image(systemName: "location.north.line.fill")
                        .font(.system(size: 18, weight: .bold))
                        .foregroundStyle(.white)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("導航中")
                        .font(.system(.caption, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.65))

                    Text(distanceString)
                        .font(.system(.headline, design: .rounded, weight: .bold))
                        .foregroundStyle(.white)
                }
            }

            HStack(spacing: 10) {
                if isUpdating {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(0.95)

                    Text("正在更新路線")
                        .font(.system(.caption, weight: .medium))
                        .foregroundStyle(.white.opacity(0.72))
                } else {
                    Text("路線會隨位置更新")
                        .font(.system(.caption, weight: .medium))
                        .foregroundStyle(.white.opacity(0.72))
                }
            }

            Button(action: onCancel) {
                Text("取消導航")
                    .font(.system(.subheadline, weight: .semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 9)
                    .background(Color.white.opacity(0.16))
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
        .padding(14)
        .frame(width: 176)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .overlay {
            RoundedRectangle(cornerRadius: 18)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        }
        .shadow(color: .black.opacity(0.28), radius: 16, y: 8)
    }

    private var distanceString: String {
        if distanceMeters < 1.5 { return "即將抵達" }
        if distanceMeters < 10 { return String(format: "%.1f 公尺", distanceMeters) }
        return String(format: "%.0f 公尺", distanceMeters)
    }
}

struct ArrivedBannerView: View {
    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 26, weight: .semibold))
                .foregroundStyle(.white)
                .symbolRenderingMode(.hierarchical)

            Text("Arrived")
                .font(.system(.title3, design: .rounded, weight: .bold))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.green.opacity(0.88))
        )
        .shadow(color: .black.opacity(0.25), radius: 10, y: 4)
    }
}

private struct CalloutPointer: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.midY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}
