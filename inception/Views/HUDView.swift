import SwiftUI

struct HUDView: View {
    let inferenceMs: Double
    let detectionCount: Int
    let orientation: AppOrientation

    var body: some View {
        HStack {
            if orientation.isFlipped {
                detectionLabel
                Spacer()
                cpuLabel
            } else {
                cpuLabel
                Spacer()
                detectionLabel
            }
        }
        // 文字翻轉 180°
        .rotationEffect(.degrees(orientation.isFlipped ? 180 : 0))
    }

    private var cpuLabel: some View {
        HUDLabel(
            icon: "cpu",
            text: String(format: "%.0f ms", inferenceMs),
            color: inferenceMs < 100 ? .green : .orange
        )
    }

    private var detectionLabel: some View {
        HUDLabel(
            icon: "eye",
            text: "\(detectionCount) 個物件",
            color: .white
        )
    }
}

struct HUDLabel: View {
    let icon: String
    let text: String
    let color: Color

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
            Text(text)
        }
        .font(.system(.caption, design: .monospaced))
        .foregroundStyle(color)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(Color.black.opacity(0.75))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .fixedSize()
    }
}
