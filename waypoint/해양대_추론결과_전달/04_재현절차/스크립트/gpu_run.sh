#!/bin/bash
# GPU 를 쓰는 스크립트를 컨테이너에서 돌린다. 인자: 스크립트 이름, 나머지는 그대로 전달
S="$1"; shift
CPP=/work/PointNeXt/openpoints/cpp
PP="$CPP/pointnet2_batch:$CPP/pointops:$CPP/chamfer_dist:$CPP/emd:/work/PointNeXt:/opt/Pointcept"
SCR='/mnt/c/Users/mando/AppData/Local/Temp/claude/c--Users-mando-Downloads---------------/88566837-f832-47bb-a614-789e138fe4a4/scratchpad'
KMOU='/mnt/c/Users/mando/Downloads/용접로봇 행동 생성 데이터/해양대_전달자료_20260810/해양대_전달자료_20260810'
RES='/mnt/c/Users/mando/Downloads/용접로봇 행동 생성 데이터/08_최종_비교실험결과/해양대66_결과'
docker run --rm --gpus all --shm-size=2g \
  -v /root/nia_check:/work -v /root/kmou_data:/data -v /root/nia_out:/out -v "$SCR":/scr:ro \
  -v "$KMOU":/kmou:ro -v "$RES":/res -v /mnt/c/Windows/Fonts:/wfonts:ro \
  -e PYTHONPATH="$PP" -e PYTHONUNBUFFERED=1 -w /opt/Pointcept \
  ptv3:run3 python "/scr/$S" "$@" 2>&1 | grep -v -e '^==*$' -e '^== CUDA' -e 'CUDA Version' \
  -e 'Container image' -e 'governed by' -e 'By pulling' -e 'developer.nvidia' -e 'NGC-DL-CONTAINER'
