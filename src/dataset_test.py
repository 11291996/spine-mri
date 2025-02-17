import os
from torch.utils.data import DataLoader
from datasets.spine_mendeley import MendeleyDataset, MendeleySAGMidDataset
from datasets.spine_gtu import GTUDataset
import cv2


def save_grayscale_png(image, save_path, bit):
    """Tensor 이미지를 8-bit PNG로 저장"""
    image_np = image.squeeze().cpu().numpy()  # [C, H, W] 또는 [H, W] → numpy 변환
    image_np = (image_np * 255).clip(0, 255).astype("uint8")
    cv2.imwrite(save_path, image_np)


def save_dicom_as_png(axis, bit=8):
    # dataset = MendeleyDataset(axis, "T1", "T2", bit=bit)
    # dataset = GTUDataset("T1", "T2", bit=bit)
    dataset = MendeleySAGMidDataset(axis, "T1", "T2", bit=bit)
    dataloader = DataLoader(dataset=dataset, pin_memory=True, batch_size=1, shuffle=False)

    print(f"Load image #: {len(dataset)}")
    print(f"Dataloader length: {len(dataloader)}")

    # 첫 번째 배치 가져오기
    data_iter = iter(dataloader)
    first_batch = next(data_iter)  # 첫 번째 배치

    # 첫 번째 배치의 두 개 리스트 (8bit/16bit 그레이스케일 PNG 이미지 리스트)
    image_list1, image_list2 = first_batch  # 두 개의 리스트

    # 중간 인덱스 찾기
    # mid_idx1 = len(image_list1) // 2
    # mid_idx2 = len(image_list2) // 2

    # # 중간 이미지 선택
    # mid_image1 = image_list1[mid_idx1]  # 첫 번째 리스트에서 중간 이미지
    # mid_image2 = image_list2[mid_idx2]  # 두 번째 리스트에서 중간 이미지

    save_grayscale_png(image_list1, f"mid_T1_{axis}_{bit}bit.png", bit)
    save_grayscale_png(image_list2, f"mid_T2_{axis}_{bit}bit.png", bit)


def main():
    save_dicom_as_png("SAG", 16)
    # save_dicom_as_png("TRA", 8)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory changed to: {script_dir}")
    main()
