"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch  # 텐서 생성과 행렬 연산 등을 제공하는 PyTorch 라이브러리.
from torch import nn  # Module, Linear 등 신경망 구성요소를 제공한다.

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):  # nn.Module을 상속하면 하위 모델·파라미터 등록, .to(), .train() 등을 사용할 수 있다.

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()  # 부모 nn.Module을 초기화한다. 하위 모듈을 self에 등록하기 전에 호출해야 한다.
        self.src_pad_idx = src_pad_idx  # src(source): 원문. 길이를 맞추기 위해 넣은 <pad> 토큰의 정수 ID.
        self.trg_pad_idx = trg_pad_idx  # trg(target): 목표 문장. 원문과 목표 언어의 어휘가 달라 패딩 ID도 따로 받는다.
        self.trg_sos_idx = trg_sos_idx  # <sos>: 문장 시작 토큰 ID. 이 클래스에서는 저장만 하고 사용하지 않는다.
        self.device = device  # 마스크를 만들 때 사용할 CPU/GPU 장치. 입력 텐서와 같은 장치여야 한다.
        self.encoder = Encoder(d_model=d_model,  # 토큰을 표현하는 벡터 차원. Encoder는 원문을 문맥 벡터로 바꾼다.
                               n_head=n_head,  # attention head 수. d_model은 n_head로 나누어떨어져야 한다.
                               max_len=max_len,  # 미리 만들 위치 인코딩의 최대 길이. 입력을 자동으로 자르는 값은 아니다.
                               ffn_hidden=ffn_hidden,  # 각 레이어의 FFN에서 중간에 확장하는 은닉 차원.
                               enc_voc_size=enc_voc_size,  # 원문 언어의 어휘 수. 원문 토큰 임베딩 테이블 크기를 결정한다.
                               drop_prob=drop_prob,  # 학습 중 dropout으로 값을 0으로 만드는 확률.
                               n_layers=n_layers,  # EncoderLayer를 순서대로 쌓는 개수. 각 층은 별도 파라미터를 가진다.
                               device=device)  # 위치 인코딩 생성 등에 사용할 장치. 전체 모델 이동은 train.py의 .to(device)가 한다.

        self.decoder = Decoder(d_model=d_model,  # Encoder 출력과 목표 문장의 앞부분을 참고해 다음 토큰의 점수를 만든다.
                               n_head=n_head,  # Decoder의 self-attention과 cross-attention에 사용할 head 수.
                               max_len=max_len,  # Decoder 위치 인코딩이 지원하는 최대 입력 길이.
                               ffn_hidden=ffn_hidden,  # Decoder FFN의 중간 은닉 차원.
                               dec_voc_size=dec_voc_size,  # 목표 언어의 어휘 수. 마지막 출력의 토큰별 점수 개수이기도 하다.
                               drop_prob=drop_prob,  # model.eval()을 호출하면 dropout은 비활성화된다.
                               n_layers=n_layers,  # DecoderLayer를 쌓는 개수. Encoder와 가중치를 공유하지 않는다.
                               device=device)  # Decoder 위치 인코딩을 생성할 장치.

    # 아래에서 B=배치 크기, S=원문 길이, L=Decoder 입력 길이, D=d_model, V=목표 어휘 수이다.
    def forward(self, src, trg):  # model(src, trg)로 호출하면 실행된다. src: [B, S], trg: [B, L]인 정수 토큰 ID.
        src_mask = self.make_src_mask(src)  # [B, 1, 1, S]: 원문의 패딩 위치를 attention에서 가린다.
        trg_mask = self.make_trg_mask(trg)  # [B, 1, L, L]: 목표 문장에 대한 패딩 조건과 미래 위치 제한을 결합한다.
        enc_src = self.encoder(src, src_mask)  # [B, S, D]: 각 원문 토큰을 주변 문맥이 반영된 벡터로 변환한다.
        output = self.decoder(trg, enc_src, trg_mask, src_mask)  # 목표 self-attention에는 trg_mask, 원문 cross-attention에는 src_mask 사용.
        return output  # [B, L, V] logits(softmax 전 점수). train.py는 trg[:, :-1]을 입력하고 trg[:, 1:]을 정답으로 사용한다.

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # 패딩이 아니면 True. [B, S] → [B, 1, S] → [B, 1, 1, S].
        # unsqueeze는 지정한 위치에 크기 1인 축을 추가한다. 데이터를 복제하지 않고 차원만 늘린다.
        # attention 점수의 축은 [B, head, query 길이, key 길이]이다. query는 정보를 찾는 위치, key는 참고할 위치다.
        # 크기 1인 head·query 축은 broadcasting으로 확장되어, 모든 head와 query가 같은 원문 패딩 key를 가린다.
        return src_mask  # 실제 attention 구현은 mask == 0인 위치의 점수를 -10000으로 바꾼 뒤 softmax를 적용한다.

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)  # [B, L] → [B, 1, L] → [B, 1, L, 1]. 패딩 query 행에 적용되는 조건.
        # 주의: 현재 코드는 패딩 key 열을 가리는 [B, 1, 1, L] 형태가 아니다. src_mask와 축의 위치가 다르다.
        trg_len = trg.shape[1]  # shape[0]은 배치 크기, shape[1]은 Decoder에 실제로 들어온 문장 길이 L이다.
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)  # [L, L] 하삼각 행렬 → uint8 → 지정 장치.
        # tril은 대각선과 그 아래만 남긴다. L=3이면 [[1, 0, 0], [1, 1, 0], [1, 1, 1]]이다.
        # 행=query, 열=key이므로 각 위치는 자신과 이전 위치만 참고한다. 정답의 미래 토큰을 미리 보는 것을 막는다.
        trg_mask = trg_pad_mask & trg_sub_mask  # &는 원소별 AND. broadcasting으로 [B, 1, L, L]이 되며 두 조건을 모두 만족해야 1이다.
        return trg_mask  # Decoder의 masked self-attention에 전달한다. 원문을 참조하는 cross-attention은 src_mask를 사용한다.
