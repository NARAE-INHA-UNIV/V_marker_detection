# JetPack 6.0 버전에 맞는 공식 이미지 사용
FROM nvcr.io/nvidia/l4t-jetpack:r36.2.0

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive

# 필수 GStreamer 플러그인 및 도구 설치
RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-rtsp \
    libgstrtspserver-1.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 컨테이너 실행 시 쉘(bash)을 기본으로 실행
# (나중에 테스트가 끝나면 아래 명령어를 스트리밍 스크립트로 바꿀 수 있습니다)
CMD ["/bin/bash"]