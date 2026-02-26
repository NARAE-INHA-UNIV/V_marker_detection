# JetPack 버전에 맞춰 베이스 이미지 선택 (JetPack 6.0 = r36.2.0)
FROM nvcr.io/nvidia/l4t-base:r36.2.0

# 환경 변수 설정 (터미널 인터랙션 방지)
ENV DEBIAN_FRONTEND=noninteractive

# 필수 GStreamer 패키지 설치 (런타임 라이브러리 포함)
RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libsoup2.4-1 \
    libgstrtspserver-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 실행 시 사용할 기본 파이프라인 스크립트 (선택 사항)
WORKDIR /app

# 컨테이너 실행 시 자동으로 스트리밍 시작
ENTRYPOINT ["gst-launch-1.0"]