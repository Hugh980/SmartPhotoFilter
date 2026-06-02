import sys

from setuptools import Distribution, setup


class Py2AppDistribution(Distribution):
    def _get_project_config_files(self, filenames):
        if filenames is None and 'py2app' in sys.argv:
            return [], []
        return super()._get_project_config_files(filenames)

APP = ['src/main.py']
DATA_FILES = []

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'assets/SmartPhotoFilter.icns',
    'packages': ['src'],
    'includes': [
        'PyQt6',
        'Vision',
        'Quartz',
        'CoreFoundation',
        'Foundation'
    ],
    'excludes': [
        'mediapipe',  # 必须排除，项目已完全迁移至 Apple Vision
        'tkinter',
        'matplotlib', # 如果后续没有用到图表，可以排除以减小体积
        'PyQt5',
        'PySide2',
        'PySide6'
    ],
    'plist': {
        'CFBundleName': 'SmartPhotoFilter',
        'CFBundleDisplayName': 'SmartPhotoFilter',
        'CFBundleIdentifier': 'com.yourname.smartphotofilter',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '11.0',
        'NSRequiresAquaSystemAppearance': False, # 支持深色/浅色模式无缝切换
        'NSHumanReadableCopyright': 'Copyright © 2026. All rights reserved.',
        # 视觉框架和文件读取可能需要的权限描述
        'NSPhotoLibraryUsageDescription': 'SmartPhotoFilter 需要读取照片以进行 AI 质量筛选。',
        'NSDesktopFolderUsageDescription': 'SmartPhotoFilter 需要访问桌面文件以读取和导出照片。',
        'NSDocumentsFolderUsageDescription': 'SmartPhotoFilter 需要访问文稿文件夹以读取和导出照片。'
    }
}

setup(
    app=APP,
    name='SmartPhotoFilter',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    distclass=Py2AppDistribution,

)
