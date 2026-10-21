"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

# 데이터 iterator와 어휘 사전 정보, conf.py의 설정값(epoch, inf 등)을 가져온다.
# data.py의 최상위 코드도 실행되므로 이때 데이터셋 로딩과 전처리가 진행된다.
from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time


def count_parameters(model):
    # numel()은 텐서의 원소 수. 기울기를 계산하는 학습 대상 파라미터만 센다.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    # weight가 있는 모듈 중 행렬 이상의 가중치에 Kaiming 균등분포 초기화를 적용한다.
    # 1차원 가중치와 bias는 여기서 변경하지 않는다.
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


# *_voc_size는 어휘 개수, d_model은 토큰의 내부 표현 차원이다.
# .to(device)는 모델 파라미터를 conf.py에서 선택한 CPU 또는 GPU로 옮긴다.
model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)  # 모델과 하위 모듈마다 초기화 함수를 호출한다.
# Adam은 기울기를 이용해 파라미터를 갱신한다. lr은 학습률,
# weight_decay는 가중치 규제 계수, eps는 수치적 안정성을 위한 작은 값이다.
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

# 검증 손실의 개선이 일정 기간 없으면 학습률에 factor를 곱해 줄인다.
# patience는 개선 없는 scheduler 호출을 얼마나 허용할지 정한다.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

# CrossEntropyLoss는 원시 점수(logits)와 정답 토큰 ID로 손실을 계산한다.
# 내부에서 log-softmax를 처리하므로 모델 출력에 softmax를 따로 적용하지 않는다.
# ignore_index에 해당하는 정답은 제외한다. 여기서는 src_pad_idx를 쓰므로
# 목표 언어의 패딩 ID(trg_pad_idx)와 같아야 목표 문장의 패딩이 올바르게 제외된다.
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()  # 학습 모드: dropout 등을 활성화한다.
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        # src: 원문 토큰 ID, trg: 정답 번역 토큰 ID. 형태는 [배치 크기, 문장 길이].
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()  # PyTorch는 기울기를 누적하므로 이전 배치의 값을 지운다.
        # Teacher forcing: 정답의 앞부분을 입력하고 한 칸 뒤의 토큰을 예측한다.
        # 예: 입력 [<sos>, 나는, 학생] → 정답 [나는, 학생, <eos>].
        output = model(src, trg[:, :-1])
        # [B, T-1, V] → [B*(T-1), V]: 모든 위치를 토큰 분류 문제로 펼친다.
        # B=배치 크기, T=정답 길이, V=목표 어휘 개수.
        # contiguous()는 view()가 가능한 연속 메모리 배치를 보장하고,
        # view()의 -1은 나머지 차원으로부터 크기를 자동 계산한다.
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        # 첫 토큰(<sos>)을 뺀 정답도 [B*(T-1)]로 펼쳐 예측 위치와 맞춘다.
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()  # 역전파: 각 파라미터에 대한 손실의 기울기를 계산한다.
        # 전체 기울기의 norm을 제한해 기울기가 너무 커지는 현상을 완화한다.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # 계산된 기울기로 실제 파라미터를 갱신한다.

        epoch_loss += loss.item()  # 스칼라 텐서를 Python 숫자로 바꿔 기록한다.
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    # 배치별 평균 손실의 산술평균이다(전체 유효 토큰 수로 가중한 평균은 아님).
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()  # 평가 모드: dropout 등을 비활성화한다.
    epoch_loss = 0
    batch_bleu = []
    # eval()과 별개로 기울기 추적을 꺼 메모리와 연산을 줄인다. 가중치는 갱신하지 않는다.
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            # 검증도 정답 앞부분을 입력한다. 모델이 스스로 문장 전체를 생성하는 방식은 아니다.
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    # 토큰 ID를 단어로 복원하고 특수 토큰을 제외한다.
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    # [T-1, V]에서 어휘 축(dim=1)의 최고 점수 토큰 ID를 선택한다.
                    # max()는 (최댓값, 인덱스)를 반환하므로 [1]이 예측 토큰 ID다.
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    # 현재 구현은 해당 문장의 BLEU 계산 중 발생한 예외를 모두 무시한다.
                    pass

            # 성공적으로 계산한 문장 점수를 배치 안에서 평균낸다.
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    # epoch는 학습 데이터 전체를 한 번 순회하는 단위다.
    # test_losses라는 이름이지만 아래에서는 검증 데이터(valid_iter)의 손실을 저장한다.
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        # 여기서 warmup은 scheduler 호출을 미루는 기준이며 학습률을 점진적으로 올리지는 않는다.
        # step은 0부터 시작하고 조건이 >이므로 warmup보다 커진 뒤 호출한다.
        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 검증 손실이 지금까지의 최솟값보다 낮을 때 모델을 저장한다.
        # state_dict()는 파라미터와 버퍼를 담으며 optimizer 상태는 포함하지 않는다.
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        # 'w' 모드로 파일을 덮어쓰되, 리스트에는 지금까지의 모든 epoch 기록이 들어 있다.
        # 저장하려면 result/와 saved/ 디렉터리가 미리 존재해야 한다.
        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        # PPL(perplexity)은 exp(손실)로 계산하며 낮을수록 좋다.
        # :.3f는 소수점 3자리, :7.3f는 최소 출력 폭 7칸에 소수점 3자리를 뜻한다.
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    # 이 파일을 직접 실행할 때 학습을 시작한다(import할 때는 이 블록을 실행하지 않는다).
    # inf = float('inf'): 양의 무한대. 첫 유한한 검증 손실이 최솟값으로 등록되게 한다.
    run(total_epoch=epoch, best_loss=inf)
