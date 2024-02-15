
# Work instruction
- reference [https://jinane.tistory.com/24](https://jinane.tistory.com/24)


# download
- [https://dicom.offis.de/en/dcmtk/dcmtk-software-development/](https://dicom.offis.de/en/dcmtk/dcmtk-software-development/)
- DCMTK 3.6.8 Source Code and Documentation (for ubuntu)

```bash
# 압축 해제
tar xvfz dcmtk-3.6.6.tar.gz
# 압축 풀어진 폴더로 이동cd dcmtk-3.6.6/
# 설치하기
cmake 
.make
make install
# 설치확인
dcmdump --version
```

