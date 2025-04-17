class ChatUI {
    constructor() {
        this.chatLog = document.getElementById('chatLog');
        this.questionInput = document.getElementById('pregunta');
        this.sendButton = document.getElementById('sendButton');
        this.statusInfo = document.getElementById('statusInfo');
        
        this.initEventListeners();
    }
    
    initEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendQuestion());
        this.questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendQuestion();
        });
    }
    
    async sendQuestion() {
        const question = this.questionInput.value.trim();
        if (!question) return;
        
        this.addMessage('user', question);
        this.questionInput.value = '';
        this.showLoading(true);
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pregunta: question })
            });
            
            if (!response.ok) {
                throw new Error(`Error ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                this.addMessage('bot', data.error, 'error');
            } else {
                const confidenceClass = this.getConfidenceClass(data.confidence);
                this.addMessage(
                    'bot', 
                    data.answer,
                    confidenceClass,
                    `Confianza: ${data.confidence} (${(data.score * 100).toFixed(1)}%)`
                );
            }
        } catch (error) {
            console.error('Error:', error);
            this.addMessage('bot', `Error: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
            this.scrollToBottom();
        }
    }
    
    addMessage(sender, text, confidenceClass = '', confidenceText = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.innerHTML = `
            <div>${text}</div>
            ${confidenceText ? `<div class="confidence ${confidenceClass}">${confidenceText}</div>` : ''}
        `;
        this.chatLog.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    showLoading(show) {
        this.sendButton.disabled = show;
        this.statusInfo.textContent = show ? 'Procesando tu pregunta...' : '';
    }
    
    scrollToBottom() {
        this.chatLog.scrollTop = this.chatLog.scrollHeight;
    }
    
    getConfidenceClass(confidence) {
        if (confidence.includes('Alta')) return 'high';
        if (confidence.includes('Media')) return 'medium';
        return 'low';
    }
}

// Initialize the chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChatUI();
});