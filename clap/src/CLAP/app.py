# CLAP: Chat Local And Persistent，网络隐私敏感场景下语境对话可加载设计实现的基于Ollama框架的本地大语言模型语义互动软件

###### 版本号：1.0.0

"""
Based on Ollama, a Graphical User Interface for Loc al Large Language Model Conversations.
"""
import base64
import importlib.metadata
import sys

import json
import pickle
import sqlite3
import re
import os
from PySide6.QtWebEngineWidgets import QWebEngineView
import chromadb
from chromadb.errors import InvalidCollectionException
import numpy as np
import itertools
import math
import ollama
import json
import pickle
import sqlite3
import sys
import re
import os
import numpy as np
import itertools
import math
import mammoth
import mistune
import chardet

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import ConnectionStyle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib import collections
from PIL import Image

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    # Backwards compatibility - importlib.metadata was added in Python 3.8
    import importlib_metadata


from datetime import datetime
from PySide6.QtGui import QAction, QFont, QGuiApplication, QKeySequence,QShortcut, QTextCursor
from PySide6.QtWidgets import QComboBox,QAbstractItemView, QHBoxLayout, QLabel, QMainWindow, QApplication, QMenu, QSizePolicy, QSplitter, QTextBrowser, QTextEdit, QWidget, QToolBar, QFileDialog, QTableView, QVBoxLayout, QHBoxLayout, QWidget, QSlider,  QGroupBox , QLabel , QWidgetAction, QPushButton, QSizePolicy
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QUrl, QVariantAnimation, Qt, QTranslator, QLocale, QLibraryInfo, QThread, Signal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PySide6.QtGui import QGuiApplication

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

from scipy.stats import gmean
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from IPython.display import display as Markdown
from tqdm.autonotebook import tqdm as notebook_tqdm
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] =  'truetype'

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)
print(current_directory)
# 改变当前工作目录
os.chdir(current_directory)



def is_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False


class ChatThread(QThread):
    new_text = Signal(str)  # 定义信号

    def __init__(self, messages, model,path ='', parent=None):
        super().__init__(parent)
        self.messages = messages
        self.model = model
        self.path = path

    def run(self):
        print('run',self.path)
        # 假设 ollama.chat 支持流式输出，返回一个迭代器
        if self.path != '':  # 如果有文件路径，则加载文件
            
            if is_image(self.path):
                # 文件是图片
                # 读取图片并转换为模型可以接受的格式
                # 这里的转换方法取决于模型的具体要求
                # 例如，将图片转换为字节流

                for response_chunk in ollama.generate(model=self.model, prompt=self.messages[-1]['content'],images = [self.path], stream=True):
                    text = response_chunk['response']
                    self.new_text.emit(text)
            elif 'pdf' in self.path:
                print('pdf')

                loader = UnstructuredPDFLoader(file_path=self.path)
                data = loader.load()

                # 对文档进行分割和处理
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                chunks = text_splitter.split_documents(data)
                
                persist_directory = "./chroma_db"

                # 创建向量数据库
                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=OllamaEmbeddings(model="mxbai-embed-large"),
                    collection_name="local-rag",
                    persist_directory=persist_directory
                )

                # 定义查询提示模板
                QUERY_PROMPT = PromptTemplate(
                    input_variables=["question"],
                    template="""You are an AI language model assistant. Your task is to generate five
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}""",
                )

                llm = ChatOllama(model=self.model)
                                
                # 创建检索器
                retriever = MultiQueryRetriever.from_llm(
                    vector_db.as_retriever(),
                    llm,
                    prompt=QUERY_PROMPT
                )

                # 定义RAG提示模板
                template = """Answer the question based ONLY on the following context:
                {context}
                Question: {question}
                """
                prompt = ChatPromptTemplate.from_template(template)

                # 构建链式调用
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                for token in chain.stream({"question": self.messages[-1]['content']}):
                    print(token, end='', flush=True)
                    self.new_text.emit(token)

            elif 'doc' in self.path:
                
                loader = UnstructuredWordDocumentLoader(file_path=self.path)
                data = loader.load()

                # 对文档进行分割和处理
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                chunks = text_splitter.split_documents(data)
                
                persist_directory = "./chroma_db"

                # 创建向量数据库
                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=OllamaEmbeddings(model="mxbai-embed-large"),
                    collection_name="local-rag",
                    persist_directory=persist_directory
                )

                # 定义查询提示模板
                QUERY_PROMPT = PromptTemplate(
                    input_variables=["question"],
                    template="""You are an AI language model assistant. Your task is to generate five
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}""",
                )

                llm = ChatOllama(model=self.model)
                                
                # 创建检索器
                retriever = MultiQueryRetriever.from_llm(
                    vector_db.as_retriever(),
                    llm,
                    prompt=QUERY_PROMPT
                )

                # 定义RAG提示模板
                template = """Answer the question based ONLY on the following context:
                {context}
                Question: {question}
                """
                prompt = ChatPromptTemplate.from_template(template)

                # 构建链式调用
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                for token in chain.stream({"question": self.messages[-1]['content']}):
                    print(token, end='', flush=True)
                    self.new_text.emit(token)

                pass

            else:
                for response_chunk in ollama.chat(model=self.model, messages=self.messages,stream=True):
                    text = response_chunk['message']['content']
                    self.new_text.emit(text)

        else:
            for response_chunk in ollama.chat(model=self.model, messages=self.messages,stream=True):
                text = response_chunk['message']['content']
                self.new_text.emit(text)
            

class ChatLocalAndPersistent(QMainWindow):
    def __init__(self):
        super().__init__()      
        self.resize(1024, 600)  # 设置窗口尺寸为1024*600  

        self.qm_files = []
        self.file_loaded_path = ''  # 加载文件路径
        # 筛选出.qm文件        
        # self.output_text_list=[]
        self.show_text = ''
        self.messages = []  

        self.init_ui()
        self.setLanguage()
        self.show()
    def init_ui(self):
        self.main_frame = QWidget()
        self.toolbar = QToolBar()   
        # 设置工具栏的文本大小
        self.toolbar.setStyleSheet("font-size: 12px")        
        self.addToolBar(self.toolbar)  
        self.translator = QTranslator(self)
        self.new_action = QAction('New Chat', self)
        self.open_action = QAction('Open Chat', self)
        self.save_action = QAction('Save Chat', self)
        self.export_action = QAction('To Markdown', self)          
        self.input_text_edit = QTextEdit()
        self.output_text_edit = QTextEdit()  
        self.file_viewer = QWebEngineView() 
        self.file_viewer.settings().setAttribute(self.file_viewer.settings().WebAttribute.PluginsEnabled, True)
        self.file_viewer.settings().setAttribute(self.file_viewer.settings().WebAttribute.PdfViewerEnabled, True)
        # self.file_viewer.load(QUrl.fromLocalFile(current_directory + "/location.jpg"))
        self.file_viewer.setUrl(QUrl())
        self.text_browser = QTextBrowser() 
        self.import_button = QPushButton("Import\nCtrl+I")       
        self.send_button = QPushButton("Send\nCtrl+Enter")                
        self.role_label = QLabel("Role", self)
        self.role_selector = QComboBox(self)
        self.model_label = QLabel("Model", self)
        self.model_selector = QComboBox(self)               
        self.memory_label = QLabel("Memory", self)
        self.memory_selector = QComboBox(self)
        self.language_label = QLabel("Language", self)
        self.language_selector = QComboBox(self)                
        self.toolbar.addAction(self.new_action)
        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.save_action)
        self.toolbar.addAction(self.export_action)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.model_label)
        self.toolbar.addWidget(self.model_selector)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.role_label)
        self.toolbar.addWidget(self.role_selector)     
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.memory_label)
        self.toolbar.addWidget(self.memory_selector)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.language_label)
        self.toolbar.addWidget(self.language_selector)    

        # 在工具栏中添加一个New action
        self.new_action.setShortcut('Ctrl+N')  # 设置快捷键为Ctrl+N
        self.new_action.triggered.connect(self.newChat)         
        # 在工具栏中添加一个Open action
        self.open_action.setShortcut('Ctrl+O')  # 设置快捷键为Ctrl+O
        self.open_action.triggered.connect(self.openChat)         
        # 在工具栏中添加一个Save action
        self.save_action.setShortcut('Ctrl+S') # 设置快捷键为Ctrl+S
        self.save_action.triggered.connect(self.saveChat)              
        # 在工具栏中添加一个Export action
        self.export_action.setShortcut('Ctrl+E') # 设置快捷键为Ctrl+E
        self.export_action.triggered.connect(self.exportMarkdown)  

        roles = ['user', 'system', 'assistant']
        self.role_selector.addItems(roles)

        memory_list = ['All', 'Input']
        self.memory_selector.addItems(memory_list)

        data = ollama.list()
        names = [model['model'] for model in data['models']]
        names.sort()
        self.model_selector.addItems(names)

        self.language_selector.currentTextChanged.connect(self.setLanguage)
        
        # 创建一个水平布局并添加表格视图和画布
        self.base_layout = QVBoxLayout()
        self.lower_layout = QHBoxLayout()
        self.upper_layout = QHBoxLayout()
        self.qm_files = [file for file in os.listdir()  if file.endswith('.qm')]
        # print(self.qm_files)
        self.language_selector.addItems(self.qm_files)
        # 创建一个新的字体对象
        font = QFont()
        font.setPointSize(12)
        # 设置字体
        self.input_text_edit.setFont(font)        
        self.input_text_edit.setAcceptDrops(True)
        self.input_text_edit.dragEnterEvent = self.dragEnterEvent
        self.input_text_edit.dropEvent = self.dropEvent

        self.output_text_edit.setFont(font)
        self.text_browser.setFont(font)
        
        # 创建一个QPushButton实例

        self.import_button.setShortcut('Ctrl+I')
        self.import_button.clicked.connect(self.importFile)
        self.import_button.setStyleSheet("font-size: 14px")
 
        self.send_button.setShortcut('Ctrl+Return') 
        self.send_button.clicked.connect(self.sendMessage)
        self.send_button.setStyleSheet("font-size: 14px")
        
        # 将文本编辑器和按钮添加到布局中
        # upper_layout.addWidget(self.output_text_edit)

        # self.upper_layout.addWidget(self.text_browser)

        splitter = QSplitter()
        splitter.addWidget(self.file_viewer)
        splitter.addWidget(self.text_browser)
        splitter.setStretchFactor(0, 1)  # 第一个部件（file_viewer）的拉伸因子为 1
        splitter.setStretchFactor(1, 4)  # 第二个部件（text_browser）的拉伸因子为 2
        self.upper_layout.addWidget(splitter)

        self.input_text_edit.setFixedHeight(100)  # 设置文本编辑框的高度为100
        self.import_button.setFixedHeight(100) 
        self.send_button.setFixedHeight(100)  # 设置按钮的高度为50
        self.lower_layout.addWidget(self.import_button)
        self.lower_layout.addWidget(self.input_text_edit)
        self.lower_layout.addWidget(self.send_button)

        self.base_layout.addLayout(self.upper_layout)
        self.base_layout.addLayout(self.lower_layout)

        # 创建一个QWidget，设置其布局为我们刚刚创建的布局，然后设置其为中心部件
        self.main_frame.setLayout(self.base_layout)
        self.setCentralWidget(self.main_frame)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_loaded_path = file_path
            self.file_viewer.load(QUrl.fromLocalFile(file_path))
            print('import file at', file_path)
    def setLanguage(self):                
        # 加载.qm文件
        self.translation = self.language_selector.currentText()
        # print(self.translation)
        self.translator.load(self.translation)
        QApplication.installTranslator(self.translator)

        self.text_labels={
                'title': QApplication.translate('Context', 'CLAP: Chat Local And Persistent. Based on Ollama, a Graphical User Interface for Local Large Language Model Conversations'),
                'new': QApplication.translate('Context', 'New Chat'),
                'open': QApplication.translate('Context', 'Open Chat'),
                'save': QApplication.translate('Context', 'Save Chat'),
                'export': QApplication.translate('Context', 'To Markdown'),
                'model': QApplication.translate('Context', 'Model'),
                'memory': QApplication.translate('Context', 'Memory'),
                'role': QApplication.translate('Context', 'Role'),
                'import': QApplication.translate('Context', 'Import')+'\nCtrl+I',
                'send': QApplication.translate('Context', 'Send')+'\nCtrl+Enter',
                'language': QApplication.translate('Context', 'Language'),
                'input_text': QApplication.translate('Context', 'Input'),
                'output_text': QApplication.translate('Context', 'Output'),
                'timestamp': QApplication.translate('Context', 'Timestamp')
            }   

        self.setWindowTitle(self.text_labels['title'])
        self.new_action.setText(self.text_labels['new'])
        self.open_action.setText(self.text_labels['open'])
        self.save_action.setText(self.text_labels['save'])
        self.export_action.setText(self.text_labels['export'])
        self.model_label.setText(self.text_labels['model'])
        self.memory_label.setText(self.text_labels['memory'])
        self.role_label.setText(self.text_labels['role'])
        self.import_button.setText(self.text_labels['import'])
        self.send_button.setText(self.text_labels['send'])
        self.language_label.setText(self.text_labels['language'])

    def resizeEvent(self, event):
        # 获取窗口的新大小
        new_width = event.size().width()
        new_height = event.size().height()
        # 调用父类的resizeEvent方法，以确保其他部件也能正确地调整大小
        super().resizeEvent(event)

    # 在主窗口类中
    def start_chat(self):
        # 获取当前的日期和时间
        now = datetime.now()
        # 将日期和时间格式化为字符串
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

        self.show_text += '\n\n'+ self.text_labels['model']+ ' '  + self.model + '\t' + self.text_labels['role'] + ' '  + self.role  +  '\t' + self.text_labels['timestamp'] + ' '  + timestamp + '\n' + self.text_labels['input_text']+ ' '   + ': ' + self.input_text + '\n' + self.text_labels['output_text']+ ' '  

        self.text_browser.setText(self.show_text) # 将文本添加到文本浏览器中
        self.chat_thread = ChatThread(self.messages, self.model, self.file_loaded_path)
        self.chat_thread.new_text.connect(self.update_text_browser)
        self.chat_thread.finished.connect(self.on_chat_finished)
        self.chat_thread.start()
    
    def on_chat_finished(self):
        # 输出完成后执行的操作

        self.show_text = self.text_browser.toPlainText() + '\n'
                       
        # output_text = self.text_browser.toPlainText()
        # self.output_text_list.append(self.show_text)
        if self.memory_selector.currentText() == 'Input':
            pass
        else:
            try:
                self.messages[-1]['content'] += 'And the Model reply is: '+ self.new_reply
            except:
                pass
        print("输出已完成")

    def update_text_browser(self, text):

        self.text_browser.moveCursor(QTextCursor.End)
        self.text_browser.insertPlainText(text)
        self.new_reply += text
        # self.text_browser.insertPlainText(text)


    def importFile(self):


        file_path, _ = QFileDialog.getOpenFileName(self, 'Import File', '', 'Image (*.jpg *.png *.jpeg);;Document (*.pdf *.doc *.docx);;Datasheet (*.csv *.xls *.xlsx);;All Files (*)')
        
        # 检查文件路径是否为空
        if file_path != '':
            self.file_loaded_path = file_path
            print('import file at', file_path)
            if is_image(file_path) or 'pdf' in file_path:            
                self.file_viewer.load(QUrl.fromLocalFile(file_path))
            elif 'csv' in file_path:
                self.df = pd.read_csv(file_path)
                html = self.df.to_html()
                self.file_viewer.setHtml(html)
            elif file_path.endswith(('.doc', '.docx')):
                with open(file_path, "rb") as doc_file:
                    result = mammoth.convert_to_html(doc_file)
                    html = result.value  # 获取生成的HTML
                    self.file_viewer.setHtml(html)

    def sendMessage(self):
        print('send message')
        self.new_reply = ''

        # 获取输入框的文本
        # input_text = self.input_text_edit.toPlainText()
        self.input_text = self.input_text_edit.toPlainText()

        # 调用Ollama的接口，获取回复文本
        # response = chat_gpt.get_response(text)

        self.model = self.model_selector.currentText()
        self.role = self.role_selector.currentText()

        if self.memory_selector.currentText() == 'Input':
            if self.messages == []:
                self.messages.append(
                {
                'role': self.role,
                'content': self.input_text,
                'tip':''
                }
                )
            else:
                self.messages.append(
                {
                'role': self.role,
                'content': self.input_text,
                'tip':'This is the history of the conversation, please do not reply to the previous messages. Remember the inpus but donot repeat the previous messages.'
                }
                )
        else:
            if self.messages == []:
                self.messages.append(
                {
                'role': self.role,
                'content': self.input_text,
                'tip':''
                }
                )
            else:
                self.messages.append(
                {
                'role': self.role,
                'content': self.input_text,
                'tip':'This is the history of the conversation, please do not reply to the previous messages. But do remember all the conversations.'
                }
                )
                
        
        self.start_chat()

        # 清空输入框
        self.input_text_edit.clear()

    def newChat(self):
        # 新建对话
        # self.output_text_list = []
        self.show_text = ''
        self.file_loaded_path = ''  # 加载文件路径
        self.messages = []
        self.output_text_edit.clear()
        self.input_text_edit.clear()
        self.file_viewer.setUrl(QUrl())
        self.text_browser.clear()
        print('new chat')
        
    def openChat(self):
        # 弹出文件对话框，获取保存文件的路径
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Chat', 'clap', 'Clap Save Files (*.clap);;All Files (*)')
        
        # 检查文件路径是否为空
        if file_path != '':
            # 以读取模式打开文件
            with open(file_path, 'rb') as f:
                # 使用pickle模块加载字典
                data = pickle.load(f)
                # 将字典的内容赋值给相应的变量
                # self.output_text_list = data['output_text_list']
                self.show_text = data['show_text']
                self.messages = data['messages']
                self.model_selector.setCurrentText(data['model'])
                self.role_selector.setCurrentText(data['role'])
                try:
                    self.file_loaded_path = data['file_loaded_path']
                except:
                    self.file_loaded_path = ''
                
                self.text_browser.setText(self.show_text) # 将文本添加到文本浏览器中
    
        # self.sendMessage()
    
    def saveChat(self):
        # 弹出文件对话框，获取保存文件的路径
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Chat', 'clap', 'Clap Save Files (*.clap);;All Files (*)')
        
        # 检查文件路径是否为空
        if file_path != '':
            # 以写入模式打开文件
            with open(file_path, 'wb') as f:
                # 创建一个字典，包含要保存的变量
                data = {
                    # 'output_text_list': self.output_text_list,
                    'show_text': self.show_text,
                    'messages': self.messages,
                    'model' : self.model_selector.currentText(),
                    'role' : self.role_selector.currentText(),
                    'file_loaded_path': self.file_loaded_path
                }
                # 使用pickle模块保存字典
                pickle.dump(data, f)

    def exportMarkdown(self):
        
        # 弹出文件对话框，获取保存文件的路径
        file_path, _ = QFileDialog.getSaveFileName(self, 'To Markdown', 'clap', 'Text Files (*.md);;All Files (*)')
        
        # 获取输出框的文本
        if (file_path != ''):
            with open(file_path, 'w', encoding='utf-8') as f:
                text_to_write = self.show_text
                f.write(text_to_write + '\n')
            print('save chat')

def main():
    # Linux desktop environments use an app's .desktop file to integrate the app
    # in to their application menus. The .desktop file of this app will include
    # the StartupWMClass key, set to app's formal name. This helps associate the
    # app's windows to its menu item.
    #
    # For association to work, any windows of the app must have WMCLASS property
    # set to match the value set in app's desktop file. For PySide6, this is set
    # with setApplicationName().

    # Find the name of the module that was used to start the app
    app_module = sys.modules["__main__"].__package__
    # Retrieve the app's metadata
    metadata = importlib.metadata.metadata(app_module)

    QApplication.setApplicationName(metadata["Formal-Name"])

    app = QApplication(sys.argv)
    main_window = ChatLocalAndPersistent()
    sys.exit(app.exec())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ChatLocalAndPersistent()
    main_window.show()  # 显示主窗口
    sys.exit(app.exec())