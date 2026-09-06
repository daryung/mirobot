#!/bin/bash
# GPU 없이 컨테이너에서 스크립트 하나를 돌린다. 인자: 스크립트 이름, 나머지는 그대로 전달
S="$1"; shift
SCR='/mnt/c/Users/mando/AppData/Local/Temp/claude/c--Users-mando-Downloads---------------/88566837-f832-47bb-a614-789e138fe4a4/scratchpad'
KMOU='/mnt/c/Users/mando/Downloads/용접로봇 행동 생성 데이터/해양대_전달자료_20260810/해양대_전달자료_20260810'
RES='/mnt/c/Users/mando/Downloads/용접로봇 행동 생성 데이터/08_최종_비교실험결과/해양대66_결과'
docker run --rm -v /root/nia_out:/out -v "$SCR":/scr:ro -v "$KMOU":/kmou:ro -v "$RES":/res \
  -v /mnt/c/Windows/Fonts:/wfonts:ro -e PYTHONUNBUFFERED=1 \
  ptv3:run3 python "/scr/$S" "$@" 2>&1 | grep -v -e '^==*$' -e '^== CUDA' -e 'CUDA Version' -e 'Container image' -e 'governed by' -e 'By pulling' -e 'developer.nvidia' -e 'NGC-DL-CONTAINER' -e 'NVIDIA Driver was not detected' -e 'NVIDIA Container Toolkit' -e 'docs.nvidia.com' -e '^$'
