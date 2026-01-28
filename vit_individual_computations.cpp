#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

// ---------------- CONFIG ----------------
const int IMG_SIZE = 32;
const int PATCH = 4;
const int NUM_PATCH = (IMG_SIZE / PATCH) * (IMG_SIZE / PATCH);
const int PATCH_DIM = PATCH * PATCH * 3;
const int D_MODEL = 64;
const int NUM_CLASSES = 2;

// ---------------- IMAGE LOADER ----------------
vector<float> read_image(const string& path, int img_size=IMG_SIZE) {
    int w,h,c;
    unsigned char* data = stbi_load(path.c_str(), &w,&h,&c,3);
    if(!data) throw runtime_error("Failed to load image: " + path);
    
    vector<float> img(img_size*img_size*3);
    for(int i=0;i<img_size*img_size*3;i++)
        img[i] = data[i]/255.0f;

    stbi_image_free(data);
    return img;
}

#include <vector>
#include <string>
using namespace std;

// Uses read_image() defined earlier
vector<vector<float>> load_batch_images(const vector<string>& paths) {
    vector<vector<float>> batch;
    for (const string& path : paths) {
        try {
            batch.push_back(read_image(path));  // read_image from earlier
        } catch (const exception& e) {
            cerr << "Failed to load image: " << path << " Error: " << e.what() << endl;
        }
    }
    return batch;
}


// ---------------- PATCHIFY ----------------
vector<vector<float>> patchify(const vector<float>& img) {
    vector<vector<float>> patches(NUM_PATCH, vector<float>(PATCH_DIM));
    int idx=0;
    for(int py=0;py<IMG_SIZE;py+=PATCH)
        for(int px=0;px<IMG_SIZE;px+=PATCH) {
            vector<float> p;
            for(int y=0;y<PATCH;y++)
                for(int x=0;x<PATCH;x++)
                    for(int c=0;c<3;c++)
                        p.push_back(img[((py+y)*IMG_SIZE+(px+x))*3+c]);
            patches[idx++] = p;
        }
    return patches;
}

// ---------------- MATMUL ----------------
vector<vector<float>> matmul_basic(const vector<vector<float>>& A,
                                   const vector<vector<float>>& B) {
    int n=A.size(), k=B.size(), m=B[0].size();
    vector<vector<float>> C(n, vector<float>(m,0));
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            for(int t=0;t<k;t++)
                C[i][j]+=A[i][t]*B[t][j];
    return C;
}

// ---------------- ATTENTION ----------------
vector<vector<float>> attention_basic(vector<vector<float>> X,
                                      const vector<vector<float>>& Wq,
                                      const vector<vector<float>>& Wk,
                                      const vector<vector<float>>& Wv) {
    auto Q = matmul_basic(X,Wq);
    auto K = matmul_basic(X,Wk);
    auto V = matmul_basic(X,Wv);

    int n=Q.size();
    int d=Q[0].size();
    vector<vector<float>> scores(n, vector<float>(n,0));

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int t=0;t<d;t++)
                scores[i][j]+=Q[i][t]*K[j][t];

    // softmax row-wise
    for(int i=0;i<n;i++){
        float sum=0;
        for(int j=0;j<n;j++){ scores[i][j]=exp(scores[i][j]); sum+=scores[i][j]; }
        for(int j=0;j<n;j++) scores[i][j]/=sum;
    }

    vector<vector<float>> out(n, vector<float>(d,0));
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int t=0;t<d;t++)
                out[i][t]+=scores[i][j]*V[j][t];
    return out;
}

// ---------------- MLP ----------------
vector<vector<float>> mlp_basic(vector<vector<float>> X,
                                const vector<vector<float>>& W1,
                                const vector<vector<float>>& W2) {
    auto H = matmul_basic(X,W1);

    // ReLU
    for(auto &row:H)
        for(auto &v:row)
            v = max(0.0f,v);

    auto out = matmul_basic(H,W2);

    // Residual
    for(int i=0;i<X.size();i++)
        for(int j=0;j<X[0].size();j++)
            X[i][j]+=out[i][j];

    return X;
}

// ---------------- TRANSFORMER BLOCK ----------------
vector<vector<float>> transformer_basic(vector<vector<float>> tokens,
                                        const vector<vector<float>>& Wq,
                                        const vector<vector<float>>& Wk,
                                        const vector<vector<float>>& Wv,
                                        const vector<vector<float>>& W1,
                                        const vector<vector<float>>& W2) {
    auto attn = attention_basic(tokens,Wq,Wk,Wv);

    for(int i=0;i<tokens.size();i++)
        for(int j=0;j<tokens[0].size();j++)
            tokens[i][j]+=attn[i][j];

    tokens = mlp_basic(tokens,W1,W2);
    return tokens;
}

// ---------------- CLASSIFIER ----------------
vector<float> classifier_forward(vector<vector<float>>& tokens,
                                 const vector<vector<float>>& Wcls) {
    vector<float> pooled = tokens[0]; // CLS token
    int C = Wcls[0].size();
    vector<float> logits(C,0);
    for(int c=0;c<C;c++)
        for(int i=0;i<pooled.size();i++)
            logits[c]+=pooled[i]*Wcls[i][c];
    return logits;
}

// ---------------- HELPER: INIT WEIGHTS ----------------
vector<vector<float>> init_weight(int in,int out,float scale=0.01f){
    vector<vector<float>> W(in, vector<float>(out, scale));
    return W;
}

// ---------------- MAIN DEMO ----------------
int main() {
    // Dummy weight matrices
    auto Wpatch = init_weight(PATCH_DIM,D_MODEL);
    auto Wq = init_weight(D_MODEL,D_MODEL);
    auto Wk = init_weight(D_MODEL,D_MODEL);
    auto Wv = init_weight(D_MODEL,D_MODEL);
    auto W1 = init_weight(D_MODEL,D_MODEL);
    auto W2 = init_weight(D_MODEL,D_MODEL);
    auto Wcls = init_weight(D_MODEL,NUM_CLASSES);

    // Dummy CLS token (add as first token later)
    vector<float> cls_token(D_MODEL,0.01f);

    // Image paths (replace with your images)
    vector<string> paths = {"cats_1.jpg","dogs_1.jpg", "cats_0.jpg", "dogs_1.jpg"};

auto batch_images = load_batch_images(paths);

for(auto &img : batch_images){
    auto patches = patchify(img);
    cout<<"Patches: "<<patches.size()<<" , first patch first value: "<<patches[0][0]<<endl;

    auto tokens = matmul_basic(patches,Wpatch);
    cout<<"Token[0] after patch embedding: "<<tokens[0][0]<<endl;

    tokens.insert(tokens.begin(), cls_token); // add CLS token
    auto attn_out = attention_basic(tokens,Wq,Wk,Wv);
    cout<<"Token[0] after attention: "<<attn_out[0][0]<<endl;

    for(int i=0;i<tokens.size();i++)
        for(int j=0;j<tokens[0].size();j++)
            tokens[i][j]+=attn_out[i][j];
    cout<<"Token[0] after residual: "<<tokens[0][0]<<endl;

    auto mlp_out = mlp_basic(tokens,W1,W2);
    cout<<"Token[0] after MLP: "<<mlp_out[0][0]<<endl;

    auto logits = classifier_forward(mlp_out,Wcls);
    cout<<"Logits: ";
    for(auto &v:logits) cout<<v<<" ";
    cout<<endl<<endl;
}

    return 0;
}
