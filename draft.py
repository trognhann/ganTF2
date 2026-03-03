import os

def fix_whitespace_names(root_directory):
    count = 0
    # Sử dụng topdown=False để đổi tên file trong thư mục trước khi đổi tên chính thư mục đó
    for root, dirs, files in os.walk(root_directory, topdown=False):
        
        # 1. Xử lý Files
        for name in files:
            if name != name.strip():
                old_path = os.path.join(root, name)
                new_name = name.strip()
                new_path = os.path.join(root, new_name)
                
                os.rename(old_path, new_path)
                print(f"Fixed File: '{name}' -> '{new_name}'")
                count += 1

        # 2. Xử lý Directories
        for name in dirs:
            if name != name.strip():
                old_path = os.path.join(root, name)
                new_name = name.strip()
                new_path = os.path.join(root, new_name)
                
                os.rename(old_path, new_path)
                print(f"Fixed Folder: '{name}' -> '{new_name}'")
                count += 1

    print(f"\n✅ Hoàn thành! Đã sửa tổng cộng {count} lỗi khoảng trắng.")

# Thay đường dẫn tới folder dataset của bạn
fix_whitespace_names('/Users/trognhann/Desktop/AnimeGANv3/dataset')