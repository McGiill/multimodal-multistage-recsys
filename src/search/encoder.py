"""CLIP Text/Image Encoder.

텍스트와 이미지를 같은 벡터 공간으로 보내고 L2 정규화해서 돌려준다.
    enc = build_encoder("openai/clip-vit-base-patch32", device="cpu")
    enc.encode_texts(["black strap top"])        # (1, 512)
    enc.encode_images([PIL.Image, ...])          # (n, 512)

모델 이름으로 백엔드를 고른다.
- transformers CLIP (기본, 명세서 조건): "openai/clip-vit-base-patch32", "patrickjohncyh/fashion-clip" 등
- open_clip (실험용, pip install open_clip_torch 필요): "hf-hub:Marqo/marqo-fashionCLIP" 처럼 hf-hub: 로 시작
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.common.faiss_utils import l2_normalize

OPEN_CLIP_PREFIX = "hf-hub:"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pad_to_square(image: Image.Image, fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """세로로 긴 상품 사진을 잘리지 않게 흰 여백으로 정사각형을 만든다."""
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _features(output: torch.Tensor | Any) -> torch.Tensor:
    """transformers 4.x는 Tensor, 5.x는 pooler_output에 투영된 특징을 담아 돌려준다."""
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


class BaseEncoder:
    """배치 처리·정규화·파일 읽기는 공통. 백엔드는 _encode_text_batch / _encode_image_batch만 구현."""

    model_name: str
    device: str
    pad_square: bool
    dim: int

    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def _encode_image_batch(self, images: list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = [
            self._encode_text_batch(list(texts[i : i + batch_size])).float().cpu().numpy()
            for i in range(0, len(texts), batch_size)
        ]
        return l2_normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), "float32")

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            batch = [pad_to_square(im) if self.pad_square else im for im in images[i : i + batch_size]]
            out.append(self._encode_image_batch(batch).float().cpu().numpy())
        return l2_normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), "float32")

    def encode_image_files(
        self, paths: Sequence[str | Path], batch_size: int = 32, num_workers: int = 8
    ) -> Iterable[np.ndarray]:
        """이미지 파일을 배치 단위로 읽어 인코딩 (파일 읽기는 스레드로 병렬 처리)."""
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for i in range(0, len(paths), batch_size):
                images = list(pool.map(load_image, paths[i : i + batch_size]))
                yield self.encode_images(images, batch_size=batch_size)


class ClipEncoder(BaseEncoder):
    """transformers CLIPModel (명세서 기본)."""

    def __init__(self, model_name: str, device: str = "cpu", pad_square: bool = True) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.pad_square = pad_square
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        # 전처리를 PIL 기반(slow)으로 고정: torchvision 설치 여부에 따라 fast 버전으로 바뀌면
        # 인덱스와 쿼리의 전처리가 미세하게 달라지고, macOS에서 OpenMP 충돌도 생긴다.
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        self.dim = int(self.model.config.projection_dim)
        # 토크나이저에 최대 길이가 없는 모델(예: fashion-clip)도 있어서 모델 설정값(77)으로 명시
        self.max_text_len = int(self.model.config.text_config.max_position_embeddings)

    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_len
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return _features(self.model.get_text_features(**inputs))

    def _encode_image_batch(self, images: list[Image.Image]) -> torch.Tensor:
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        return _features(self.model.get_image_features(pixel_values=pixel_values))


class OpenClipEncoder(BaseEncoder):
    """open_clip 모델 (예: Marqo-FashionCLIP). 실험용."""

    def __init__(self, model_name: str, device: str = "cpu", pad_square: bool = True) -> None:
        import open_clip

        self.model_name = model_name
        self.device = resolve_device(device)
        self.pad_square = pad_square
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.dim = int(self.model.text_projection.shape[1])

    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode_text(self.tokenizer(texts).to(self.device))

    def _encode_image_batch(self, images: list[Image.Image]) -> torch.Tensor:
        pixels = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        return self.model.encode_image(pixels)


def build_encoder(model_name: str, device: str = "cpu", pad_square: bool = True) -> BaseEncoder:
    cls = OpenClipEncoder if model_name.startswith(OPEN_CLIP_PREFIX) else ClipEncoder
    return cls(model_name, device=device, pad_square=pad_square)
